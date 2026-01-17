//! # proj
//!
//! Symbolic projection (embeddings) for Tekne.
//!
//! Maps discrete symbols to continuous vectors using a Codebook.
//!
//! ## Intuition First
//!
//! Imagine a library where every book has a call number. The call number
//! isn't just a label; it tells you where the book sits in a 3D space.
//! `proj` is the system that maps "book names" (tokens) to "library coordinates" (vectors).

use innr::pool_mean;
use textprep::SubwordTokenizer;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Token not found in codebook: {0}")]
    TokenNotFound(u32),
    #[error("Encoding error: {0}")]
    Encoding(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A Codebook maps token IDs to dense vectors.
pub struct Codebook {
    /// Flattened embedding matrix [vocab_size * dim]
    matrix: Vec<f32>,
    /// Dimension of each vector
    dim: usize,
}

impl Codebook {
    /// Create a new Codebook from a flattened matrix and dimension.
    pub fn new(matrix: Vec<f32>, dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Encoding("Dimension cannot be zero".to_string()));
        }
        if matrix.len() % dim != 0 {
            return Err(Error::Encoding("Matrix size must be multiple of dimension".to_string()));
        }
        Ok(Self { matrix, dim })
    }

    /// Get the vector for a token ID.
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        let start = (id as usize) * self.dim;
        let end = start + self.dim;
        if end <= self.matrix.len() {
            Some(&self.matrix[start..end])
        } else {
            None
        }
    }

    /// Get the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// A Projection combines a Tokenizer and a Codebook.
pub struct Projection<T: SubwordTokenizer> {
    tokenizer: T,
    codebook: Codebook,
}

impl<T: SubwordTokenizer> Projection<T> {
    /// Create a new Projection.
    pub fn new(tokenizer: T, codebook: Codebook) -> Self {
        Self { tokenizer, codebook }
    }

    /// Encode text into a single vector using mean pooling.
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.tokenize(text);
        if tokens.is_empty() {
            return Ok(vec![0.0; self.codebook.dim()]);
        }

        let embeddings: Vec<&[f32]> = tokens.iter()
            .filter_map(|&id| self.codebook.get(id))
            .collect();

        if embeddings.is_empty() {
             return Ok(vec![0.0; self.codebook.dim()]);
        }

        let mut out = vec![0.0; self.codebook.dim()];
        pool_mean(&embeddings, &mut out);
        Ok(out)
    }

    /// Encode text into a sequence of vectors (no pooling).
    pub fn encode_sequence(&self, text: &str) -> Result<Vec<Vec<f32>>> {
        let tokens = self.tokenizer.tokenize(text);
        let mut result = Vec::with_capacity(tokens.len());
        
        for id in tokens {
            if let Some(emb) = self.codebook.get(id) {
                result.push(emb.to_vec());
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use textprep::BpeTokenizer;
    use std::collections::HashMap;

    #[test]
    fn test_projection_basic() {
        let mut vocab = HashMap::new();
        vocab.insert("apple".to_string(), 0);
        vocab.insert("pie".to_string(), 1);
        let tokenizer = BpeTokenizer::from_vocab(vocab);

        let matrix = vec![
            1.0, 0.0, 0.0, // apple
            0.0, 1.0, 0.0, // pie
        ];
        let codebook = Codebook::new(matrix, 3).unwrap();
        let proj = Projection::new(tokenizer, codebook);

        let vec = proj.encode("apple pie").unwrap();
        // Mean pooling: ( [1,0,0] + [0,1,0] ) / 2 = [0.5, 0.5, 0]
        assert!((vec[0] - 0.5).abs() < 1e-6);
        assert!((vec[1] - 0.5).abs() < 1e-6);
        assert!((vec[2] - 0.0).abs() < 1e-6);
    }
}
