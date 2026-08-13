//! # symproj
//!
//! Codebook-based token-to-vector projection.
//!
//! Maps token IDs to dense vectors with a [`Codebook`], then exposes pooled
//! phrase vectors and per-token vector sequences.
//!
//! ## Scope
//!
//! `symproj` does not train embedding models or load external embedding file
//! formats. Use it when the vocabulary and embedding matrix already exist and
//! the remaining job is lookup, pooling, normalization, or sequence output for
//! downstream retrieval code.

use std::{ops::Index, sync::Arc};

use textprep::SubwordTokenizer;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Token not found in codebook: {0}")]
    TokenNotFound(u32),
    #[error("Weight length mismatch: expected {expected}, got {got}")]
    WeightLenMismatch { expected: usize, got: usize },
    #[error("dimension cannot be zero")]
    ZeroDimension,
    #[error("matrix length {len} is not a multiple of dimension {dim}")]
    InvalidMatrixShape { len: usize, dim: usize },
    #[error("weight at index {index} must be finite")]
    NonFiniteWeight { index: usize },
    #[error("component direction must have finite unit norm")]
    InvalidUnitDirection,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Counts how many token IDs contributed to a lenient encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EncodeStats {
    /// Number of token IDs passed by the caller.
    pub total: usize,
    /// Number of token IDs found in the codebook.
    pub present: usize,
    /// Token IDs that were not present in the codebook, preserving input order.
    pub missing_ids: Vec<u32>,
}

impl EncodeStats {
    fn new(total: usize) -> Self {
        Self {
            total,
            present: 0,
            missing_ids: Vec::new(),
        }
    }

    /// Number of token IDs skipped because they were not in the codebook.
    #[must_use]
    pub fn missing(&self) -> usize {
        self.missing_ids.len()
    }

    /// Whether every input token ID was present in the codebook.
    #[must_use]
    pub fn all_present(&self) -> bool {
        self.missing_ids.is_empty()
    }
}

/// A Codebook maps token IDs to dense vectors.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Flattened embedding matrix, laid out as `vocab_size * dim`.
    matrix: Arc<[f32]>,
    /// Dimension of each vector
    dim: usize,
}

impl Codebook {
    /// Create a new Codebook from a flattened matrix and dimension.
    pub fn new(matrix: Vec<f32>, dim: usize) -> Result<Self> {
        Self::from_arc(Arc::from(matrix.into_boxed_slice()), dim)
    }

    /// Create a new Codebook from shared flattened matrix storage.
    pub fn from_arc(matrix: Arc<[f32]>, dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::ZeroDimension);
        }
        if !matrix.len().is_multiple_of(dim) {
            return Err(Error::InvalidMatrixShape {
                len: matrix.len(),
                dim,
            });
        }
        Ok(Self { matrix, dim })
    }

    /// Get the vector for a token ID.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        let start = (id as usize).checked_mul(self.dim)?;
        let end = start.checked_add(self.dim)?;
        if end <= self.matrix.len() {
            Some(&self.matrix[start..end])
        } else {
            None
        }
    }

    /// The raw flattened embedding matrix `[vocab_size * dim]`.
    #[must_use]
    pub fn matrix(&self) -> &[f32] {
        &self.matrix
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of token IDs representable by this codebook.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.matrix.len() / self.dim
    }

    /// Iterate over all token vectors in token-ID order.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[f32]> + '_ {
        self.matrix.chunks_exact(self.dim)
    }
}

impl Codebook {
    /// Encode a token-id sequence into a single vector using mean pooling.
    ///
    /// This is **lenient**: token IDs not present in the codebook are skipped.
    #[must_use]
    pub fn encode_ids(&self, ids: &[u32]) -> Vec<f32> {
        let mut out = vec![0.0; self.dim];
        self.encode_ids_into(ids, &mut out)
            .expect("fresh output vector has codebook dimension");
        out
    }

    /// Encode token IDs with lenient mean pooling and return skip counts.
    ///
    /// Missing IDs are skipped, as in [`Self::encode_ids`], but the returned
    /// [`EncodeStats`] lets callers distinguish empty input from all-missing
    /// input and monitor vocabulary drift.
    #[must_use]
    pub fn encode_ids_with_stats(&self, ids: &[u32]) -> (Vec<f32>, EncodeStats) {
        let mut out = vec![0.0; self.dim];
        let stats = self
            .encode_ids_into(ids, &mut out)
            .expect("fresh output vector has codebook dimension");
        (out, stats)
    }

    /// Encode token IDs into a caller-provided output buffer.
    ///
    /// The output buffer is overwritten with the lenient mean-pooled vector.
    /// Missing token IDs are skipped and reported in the returned stats.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when `out.len() != self.dim()`.
    pub fn encode_ids_into(&self, ids: &[u32], out: &mut [f32]) -> Result<EncodeStats> {
        if out.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: out.len(),
            });
        }

        out.fill(0.0);
        let mut stats = EncodeStats::new(ids.len());
        for &id in ids {
            if let Some(emb) = self.get(id) {
                stats.present += 1;
                for (o, &e) in out.iter_mut().zip(emb.iter()) {
                    *o += e;
                }
            } else {
                stats.missing_ids.push(id);
            }
        }

        if stats.present > 0 {
            let inv = 1.0 / stats.present as f32;
            for o in out.iter_mut() {
                *o *= inv;
            }
        }

        Ok(stats)
    }

    /// Encode token IDs into a single vector using mean pooling (strict).
    ///
    /// Unlike [`Self::encode_ids`], this returns an error if any token ID is not present in the
    /// codebook. This is useful when you need a “closed vocabulary” contract.
    pub fn encode_ids_strict(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.dim];
        for &id in ids {
            let emb = self.get(id).ok_or(Error::TokenNotFound(id))?;
            for (o, &e) in out.iter_mut().zip(emb.iter()) {
                *o += e;
            }
        }
        if !ids.is_empty() {
            let inv = 1.0 / ids.len() as f32;
            for o in out.iter_mut() {
                *o *= inv;
            }
        }
        Ok(out)
    }

    /// Encode token IDs into a single vector using a weighted mean (strict).
    ///
    /// We compute the weighted mean:
    ///
    /// ```text
    /// v = sum_i(w_i * E[t_i]) / sum_i(w_i)
    /// ```
    ///
    /// If `sum_i(w_i) <= 0`, this returns the zero vector.
    ///
    /// This is a weighted mean, so the denominator is `sum_w`. The original
    /// SIF sentence embedding from Arora et al. (2017) divides by sentence
    /// length before first-principal-component removal; callers that need that
    /// exact convention should scale weights before calling this method.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WeightLenMismatch`] when the lengths differ,
    /// [`Error::TokenNotFound`] when an ID is absent, or
    /// [`Error::NonFiniteWeight`] when a weight is NaN or infinite.
    pub fn encode_ids_weighted_strict(&self, ids: &[u32], weights: &[f32]) -> Result<Vec<f32>> {
        if ids.len() != weights.len() {
            return Err(Error::WeightLenMismatch {
                expected: ids.len(),
                got: weights.len(),
            });
        }
        if ids.is_empty() {
            return Ok(vec![0.0; self.dim]);
        }
        if let Some(index) = weights.iter().position(|weight| !weight.is_finite()) {
            return Err(Error::NonFiniteWeight { index });
        }

        let dim = self.dim;
        let mut out = vec![0.0f32; dim];
        let mut sum_w = 0.0f32;

        for (&id, &w) in ids.iter().zip(weights.iter()) {
            let emb = self.get(id).ok_or(Error::TokenNotFound(id))?;
            if w == 0.0 {
                continue;
            }
            sum_w += w;
            for (o, &e) in out.iter_mut().zip(emb.iter()) {
                *o = w.mul_add(e, *o);
            }
        }

        if sum_w <= 0.0 {
            return Ok(vec![0.0; dim]);
        }

        for o in out.iter_mut() {
            *o /= sum_w;
        }
        Ok(out)
    }

    /// Encode token IDs into per-token vectors (no pooling).
    ///
    /// Returns one vector per token ID that is present in the codebook.
    /// Token IDs missing from the codebook are silently skipped (same
    /// lenient behaviour as [`Self::encode_ids`]).
    ///
    /// ## `ColBERT` Late-Interaction Workflow
    ///
    /// This method produces the per-token vector sequence used in the
    /// `ColBERT` retrieval model (Khattab & Zaharia, 2020). `ColBERT` avoids
    /// pooling: every token in the query and every token in a document gets
    /// its own vector, and scoring is done via `MaxSim` (maximum inner product
    /// between each query token and all document tokens).
    ///
    /// ```rust
    /// use symproj::Codebook;
    ///
    /// // Tiny codebook: 4 tokens, 3-D vectors
    /// let matrix = vec![
    ///     1.0, 0.0, 0.0,   // token 0: "the"
    ///     0.0, 1.0, 0.0,   // token 1: "cat"
    ///     0.0, 0.0, 1.0,   // token 2: "sat"
    ///     0.5, 0.5, 0.0,   // token 3: "mat"
    /// ];
    /// let codebook = Codebook::new(matrix, 3).unwrap();
    ///
    /// // Query tokens: "the cat"
    /// let query_vecs = codebook.encode_sequence_ids(&[0, 1]);
    /// assert_eq!(query_vecs.len(), 2);
    ///
    /// // Document tokens: "cat sat mat"
    /// let doc_vecs = codebook.encode_sequence_ids(&[1, 2, 3]);
    /// assert_eq!(doc_vecs.len(), 3);
    ///
    /// // MaxSim: for each query token, max inner product over all doc tokens.
    /// // Then sum over query tokens for the ColBERT score.
    /// // In production, pipe into `rankops::rerank::colbert` for MaxSim scoring.
    /// let mut colbert_score = 0.0f32;
    /// for q in &query_vecs {
    ///     let max_sim: f32 = doc_vecs.iter()
    ///         .map(|d| q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>())
    ///         .fold(f32::NEG_INFINITY, f32::max);
    ///     colbert_score += max_sim;
    /// }
    /// assert!(colbert_score > 0.0);
    /// ```
    ///
    /// For large-scale use, pass the output of this method directly to
    /// `rankops::rerank::colbert`, which implements the efficient `MaxSim`
    /// computation over a candidate set.
    #[must_use]
    pub fn encode_sequence_ids(&self, ids: &[u32]) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(emb) = self.get(id) {
                result.push(emb.to_vec());
            }
        }
        result
    }

    /// Encode token IDs into one flat, row-major token-vector buffer.
    ///
    /// The returned `usize` is the number of present token IDs. Each present
    /// token occupies one contiguous `self.dim()` row in the returned vector.
    /// Missing token IDs are skipped, matching [`Self::encode_sequence_ids`].
    #[must_use]
    pub fn encode_sequence_ids_flat(&self, ids: &[u32]) -> (Vec<f32>, usize) {
        let mut out = Vec::with_capacity(ids.len() * self.dim);
        let mut present = 0;
        for &id in ids {
            if let Some(emb) = self.get(id) {
                out.extend_from_slice(emb);
                present += 1;
            }
        }
        (out, present)
    }
}

impl Index<u32> for Codebook {
    type Output = [f32];

    fn index(&self, index: u32) -> &Self::Output {
        self.get(index).expect("token ID out of codebook bounds")
    }
}

/// SIF (Smooth Inverse Frequency) weight from Arora et al. (2017):
/// \[
/// w(p) = \frac{a}{a + p}
/// \]
/// where \(p\) is token probability and \(a\) is a small smoothing constant (often \(10^{-3}\)).
///
/// Returns `0.0` when either input is non-finite, `a <= 0`, or `p < 0`.
/// Negative probabilities are not meaningful, so this function treats them as a no-op.
#[inline]
#[must_use]
pub fn sif_weight(p: f32, a: f32) -> f32 {
    if !p.is_finite() || !a.is_finite() || a <= 0.0 || p < 0.0 {
        return 0.0;
    }
    a / (a + p)
}

/// L2-normalize a vector in place.
///
/// If the input has norm 0, this is a no-op.
pub fn l2_normalize_in_place(v: &mut [f32]) {
    #[cfg(feature = "simd")]
    {
        innr::normalize(v);
    }

    #[cfg(not(feature = "simd"))]
    {
        const NORM_EPSILON: f32 = 1e-9;
        let mut ss = 0.0f32;
        for &x in v.iter() {
            ss += x * x;
        }
        let norm = ss.sqrt();
        if norm <= NORM_EPSILON {
            return;
        }
        let inv = 1.0f32 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Remove a component direction \(u\) from a vector \(v\):
/// \[
/// v \leftarrow v - u \frac{u \cdot v}{u \cdot u}
/// \]
///
/// This is the “remove top PC” post-step used in SIF-style baselines (when \(u\) is the top PC).
/// A zero direction is treated as a no-op.
///
/// # Errors
///
/// Returns [`Error::DimensionMismatch`] when `v` and `u` have different lengths.
pub fn remove_component_in_place(v: &mut [f32], u: &[f32]) -> Result<()> {
    if v.len() != u.len() {
        return Err(Error::DimensionMismatch {
            expected: v.len(),
            got: u.len(),
        });
    }

    let mut dot = 0.0f32;
    let mut uu = 0.0f32;
    for (&ui, &vi) in u.iter().zip(v.iter()) {
        dot = ui.mul_add(vi, dot);
        uu = ui.mul_add(ui, uu);
    }
    if uu <= 0.0 {
        return Ok(());
    }

    let scale = dot / uu;
    for (vi, &ui) in v.iter_mut().zip(u.iter()) {
        *vi -= ui * scale;
    }
    Ok(())
}

/// Remove a pre-normalized component direction from a vector.
///
/// This skips the `u · u` denominator used by [`remove_component_in_place`].
/// Use it only when the caller already knows `u_unit` has unit norm.
///
/// # Errors
///
/// Returns [`Error::DimensionMismatch`] when `v` and `u_unit` have different lengths,
/// or [`Error::InvalidUnitDirection`] when `u_unit` is non-finite or does not have unit norm.
pub fn remove_component_unit_in_place(v: &mut [f32], u_unit: &[f32]) -> Result<()> {
    if v.len() != u_unit.len() {
        return Err(Error::DimensionMismatch {
            expected: v.len(),
            got: u_unit.len(),
        });
    }
    let norm = u_unit.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || (norm - 1.0).abs() >= 0.01 {
        return Err(Error::InvalidUnitDirection);
    }
    let mut dot = 0.0f32;
    for (&ui, &vi) in u_unit.iter().zip(v.iter()) {
        dot = ui.mul_add(vi, dot);
    }
    for (vi, &ui) in v.iter_mut().zip(u_unit.iter()) {
        *vi -= ui * dot;
    }
    Ok(())
}

/// A Projection combines a Tokenizer and a Codebook.
pub struct Projection<T: SubwordTokenizer> {
    tokenizer: T,
    codebook: Codebook,
}

impl<T: SubwordTokenizer> Projection<T> {
    /// Create a new Projection.
    pub fn new(tokenizer: T, codebook: Codebook) -> Self {
        Self {
            tokenizer,
            codebook,
        }
    }

    /// Encode text into a single vector using mean pooling.
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use symproj::{Codebook, Projection};
    /// use textprep::VocabTokenizer;
    ///
    /// let mut vocab = HashMap::new();
    /// vocab.insert("hello".to_string(), 0);
    /// vocab.insert("world".to_string(), 1);
    ///
    /// let tokenizer = VocabTokenizer::from_vocab(vocab);
    /// let codebook = Codebook::new(vec![
    ///     1.0, 0.0, // hello
    ///     0.0, 1.0, // world
    /// ], 2).unwrap();
    /// let projection = Projection::new(tokenizer, codebook);
    ///
    /// assert_eq!(projection.encode("hello world"), vec![0.5, 0.5]);
    /// ```
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<f32> {
        let tokens = self.tokenizer.tokenize(text);
        self.codebook.encode_ids(&tokens)
    }

    /// Encode text into a mean-pooled vector and return coverage stats for the
    /// token IDs emitted by the tokenizer.
    #[must_use]
    pub fn encode_with_stats(&self, text: &str) -> (Vec<f32>, EncodeStats) {
        let tokens = self.tokenizer.tokenize(text);
        self.codebook.encode_ids_with_stats(&tokens)
    }

    /// Encode text into a sequence of vectors (no pooling).
    #[must_use]
    pub fn encode_sequence(&self, text: &str) -> Vec<Vec<f32>> {
        let tokens = self.tokenizer.tokenize(text);
        self.codebook.encode_sequence_ids(&tokens)
    }

    /// Encode text into one flat, row-major token-vector buffer.
    #[must_use]
    pub fn encode_sequence_flat(&self, text: &str) -> (Vec<f32>, usize) {
        let tokens = self.tokenizer.tokenize(text);
        self.codebook.encode_sequence_ids_flat(&tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap, sync::Arc};
    use textprep::VocabTokenizer;

    static_assertions::assert_impl_all!(Codebook: Send, Sync);

    #[derive(Clone)]
    struct FixedTokenizer(Vec<u32>);

    impl SubwordTokenizer for FixedTokenizer {
        fn tokenize(&self, _text: &str) -> Vec<u32> {
            self.0.clone()
        }
    }

    #[test]
    fn test_projection_basic() {
        let mut vocab = HashMap::new();
        vocab.insert("apple".to_string(), 0);
        vocab.insert("pie".to_string(), 1);
        let tokenizer = VocabTokenizer::from_vocab(vocab);

        let matrix = vec![
            1.0, 0.0, 0.0, // apple
            0.0, 1.0, 0.0, // pie
        ];
        let codebook = Codebook::new(matrix, 3).unwrap();
        let proj = Projection::new(tokenizer, codebook);

        let vec = proj.encode("apple pie");
        // Mean pooling: ( [1,0,0] + [0,1,0] ) / 2 = [0.5, 0.5, 0]
        assert!((vec[0] - 0.5).abs() < 1e-6);
        assert!((vec[1] - 0.5).abs() < 1e-6);
        assert!((vec[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_codebook_rejects_zero_dim() {
        let err = Codebook::new(vec![1.0, 2.0, 3.0], 0).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("dimension cannot be zero"), "got: {msg}");
    }

    #[test]
    fn test_codebook_rejects_non_multiple() {
        let err = Codebook::new(vec![1.0, 2.0, 3.0], 2).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not a multiple of dimension"), "got: {msg}");
    }

    #[test]
    fn codebook_from_arc_shares_matrix_storage() {
        let matrix = Arc::<[f32]>::from([1.0, 2.0, 3.0, 4.0]);
        let codebook = Codebook::from_arc(Arc::clone(&matrix), 2).unwrap();

        assert_eq!(Arc::strong_count(&matrix), 2);
        assert_eq!(codebook.matrix(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn codebook_iter_yields_vectors_in_token_order() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0, 4.0], 2).unwrap();
        let rows: Vec<&[f32]> = codebook.iter().collect();

        assert_eq!(rows, vec![&[1.0, 2.0][..], &[3.0, 4.0][..]]);
        assert_eq!(&codebook[1], &[3.0, 4.0]);
    }

    #[test]
    fn codebook_strict_errors_on_missing_token() {
        let codebook = Codebook::new(vec![1.0, 2.0], 2).unwrap(); // vocab_size=1
        let err = codebook.encode_ids_strict(&[0, 9]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Token not found"), "got: {msg}");
    }

    #[test]
    fn weighted_mean_matches_unweighted_mean_when_all_weights_equal() {
        let matrix = vec![
            1.0, 0.0, // id=0
            0.0, 1.0, // id=1
        ];
        let codebook = Codebook::new(matrix, 2).unwrap();
        let ids = [0u32, 1u32];
        let w = [1.0f32, 1.0f32];
        let v = codebook.encode_ids_weighted_strict(&ids, &w).unwrap();
        assert!((v[0] - 0.5).abs() < 1e-6);
        assert!((v[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_has_unit_norm_when_nonzero() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize_in_place(&mut v);
        let norm = v[0].hypot(v[1]);
        assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");
    }

    #[test]
    fn single_token_equals_embedding() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0], 3).unwrap();
        let v = codebook.encode_ids(&[0]);
        assert_eq!(&v[..], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn encode_ids_all_missing_returns_zero_vector() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0], 3).unwrap();
        let v = codebook.encode_ids(&[999]);
        assert_eq!(&v[..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn encode_ids_with_stats_distinguishes_empty_from_all_missing() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0], 3).unwrap();

        let (empty, empty_stats) = codebook.encode_ids_with_stats(&[]);
        assert_eq!(&empty[..], &[0.0, 0.0, 0.0]);
        assert_eq!(empty_stats.total, 0);
        assert_eq!(empty_stats.present, 0);
        assert_eq!(empty_stats.missing(), 0);
        assert!(empty_stats.all_present());

        let (missing, missing_stats) = codebook.encode_ids_with_stats(&[999, 1000]);
        assert_eq!(&missing[..], &[0.0, 0.0, 0.0]);
        assert_eq!(missing_stats.total, 2);
        assert_eq!(missing_stats.present, 0);
        assert_eq!(missing_stats.missing_ids, vec![999, 1000]);
        assert!(!missing_stats.all_present());
    }

    #[test]
    fn encode_ids_into_reports_missing_and_reuses_output() {
        let codebook = Codebook::new(vec![1.0, 3.0, 5.0, 7.0], 2).unwrap();
        let mut out = vec![99.0, 99.0];

        let stats = codebook.encode_ids_into(&[0, 9, 1], &mut out).unwrap();

        assert_eq!(&out[..], &[3.0, 5.0]);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.present, 2);
        assert_eq!(stats.missing_ids, vec![9]);
    }

    #[test]
    fn encode_ids_into_rejects_wrong_output_dimension() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0], 3).unwrap();
        let mut out = vec![0.0, 0.0];

        let err = codebook.encode_ids_into(&[0], &mut out).unwrap_err();

        assert!(matches!(
            err,
            Error::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn encode_sequence_ids_flat_matches_nested_sequence_output() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0, 4.0], 2).unwrap();

        let nested = codebook.encode_sequence_ids(&[0, 9, 1]);
        let (flat, present) = codebook.encode_sequence_ids_flat(&[0, 9, 1]);

        assert_eq!(nested, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(present, 2);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn weighted_zero_weights_returns_zero_vector() {
        let codebook = Codebook::new(vec![1.0, 2.0, 3.0], 3).unwrap();
        let v = codebook.encode_ids_weighted_strict(&[0], &[0.0]).unwrap();
        assert_eq!(&v[..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn weighted_mean_rejects_nonfinite_weights() {
        let codebook = Codebook::new(vec![1.0, 0.0, 0.0, 1.0], 2).unwrap();
        for weight in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = codebook
                .encode_ids_weighted_strict(&[0, 1], &[weight, 1.0])
                .unwrap_err();
            assert!(matches!(err, Error::NonFiniteWeight { index: 0 }));
        }
    }

    #[test]
    fn weighted_mean_preserves_finite_signed_weight_behavior() {
        let codebook = Codebook::new(vec![1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let encoded = codebook
            .encode_ids_weighted_strict(&[0, 1], &[-1.0, 2.0])
            .unwrap();

        assert_eq!(encoded, vec![-1.0, 2.0]);
    }

    #[test]
    fn l2_normalize_noop_on_zero_vector() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize_in_place(&mut v);
        assert_eq!(&v[..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_noop_below_shared_norm_threshold() {
        let mut v = vec![1e-10f32, 0.0];
        l2_normalize_in_place(&mut v);
        assert_eq!(v, vec![1e-10, 0.0]);
    }

    #[test]
    fn sif_weight_invalid_inputs_return_zero() {
        for (p, a) in [
            (-0.1, 1e-3),
            (f32::NAN, 1e-3),
            (f32::INFINITY, 1e-3),
            (0.1, f32::NAN),
            (0.1, f32::INFINITY),
            (0.1, f32::NEG_INFINITY),
        ] {
            assert_eq!(sif_weight(p, a), 0.0);
        }
    }

    #[test]
    fn remove_component_dimension_mismatch_errors() {
        let mut v = vec![1.0f32, 2.0];
        let u = vec![0.0f32, 1.0, 0.0];
        assert!(remove_component_in_place(&mut v, &u).is_err());
    }

    #[test]
    fn remove_component_handles_non_unit_direction() {
        let mut v = vec![3.0f32, 4.0, 0.0];
        let u = vec![2.0f32, 0.0, 0.0];

        remove_component_in_place(&mut v, &u).unwrap();

        let dot: f32 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-6, "dot={dot}");
        assert_eq!(&v[..], &[0.0, 4.0, 0.0]);
    }

    #[test]
    fn remove_component_unit_fast_path_matches_arbitrary_direction_path() {
        let mut via_general = vec![3.0f32, 4.0, 0.0];
        let mut via_unit = via_general.clone();
        let u = vec![1.0f32, 0.0, 0.0];

        remove_component_in_place(&mut via_general, &u).unwrap();
        remove_component_unit_in_place(&mut via_unit, &u).unwrap();

        assert_eq!(via_unit, via_general);
    }

    #[test]
    fn remove_component_unit_rejects_invalid_direction_without_mutating() {
        for u in [vec![2.0f32, 0.0], vec![0.0, 0.0], vec![f32::NAN, 0.0]] {
            let mut v = vec![3.0f32, 4.0];
            let before = v.clone();

            let err = remove_component_unit_in_place(&mut v, &u).unwrap_err();

            assert!(matches!(err, Error::InvalidUnitDirection));
            assert_eq!(v, before);
        }
    }

    #[test]
    fn remove_component_is_idempotent_and_direction_scale_invariant() {
        let original = vec![3.0f32, 4.0, 5.0];
        let direction = vec![1.0f32, -2.0, 0.5];
        let scaled_direction: Vec<f32> = direction.iter().map(|x| x * -3.0).collect();
        let mut projected = original.clone();
        let mut projected_scaled = original;

        remove_component_in_place(&mut projected, &direction).unwrap();
        remove_component_in_place(&mut projected_scaled, &scaled_direction).unwrap();
        for (&a, &b) in projected.iter().zip(&projected_scaled) {
            assert!((a - b).abs() < 1e-5, "scale invariance: {a} != {b}");
        }

        let once = projected.clone();
        remove_component_in_place(&mut projected, &direction).unwrap();
        for (&a, &b) in projected.iter().zip(&once) {
            assert!((a - b).abs() < 1e-5, "idempotence: {a} != {b}");
        }
    }

    #[test]
    fn projection_encode_with_stats_reports_token_coverage() {
        let tokenizer = FixedTokenizer(vec![0, 9]);
        let codebook = Codebook::new(vec![2.0, 4.0], 2).unwrap();
        let proj = Projection::new(tokenizer, codebook);

        let (vec, stats) = proj.encode_with_stats("ignored by fixed tokenizer");

        assert_eq!(&vec[..], &[2.0, 4.0]);
        assert_eq!(stats.total, 2);
        assert_eq!(stats.present, 1);
        assert_eq!(stats.missing_ids, vec![9]);
    }

    #[test]
    fn multilingual_vocab_smoke() {
        // Minimal multilingual coverage (scripts + diacritics).
        let mut vocab = HashMap::new();
        vocab.insert("東京".to_string(), 0);
        vocab.insert("Москва".to_string(), 1);
        vocab.insert("التقى".to_string(), 2);
        vocab.insert("राम".to_string(), 3);
        vocab.insert("François".to_string(), 4);
        let tokenizer = VocabTokenizer::from_vocab(vocab);

        // 5 tokens, 1-D vectors for simplicity.
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let codebook = Codebook::new(matrix, 1).unwrap();
        let proj = Projection::new(tokenizer, codebook);

        let v = proj.encode("東京 Москва التقى राम François");
        assert!((v[0] - 3.0).abs() < 1e-6, "got={:?}", v);
    }
}
