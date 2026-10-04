//! Parallel fills. Rows depend only on the key and the position, so whole rows fill
//! independently and the result is the serial fill.

use rayon::prelude::*;

use crate::{Elem, Tandem};

/// Rows per task: enough work to hide the reseed of a task's first group.
const TASK_ROWS: usize = 1024;

impl Tandem {
    fn par_fill<T: Elem + Send>(&mut self, out: &mut [T]) {
        let w = u64::from(T::BITS);
        let per_row = (1024 / T::BITS) as usize;
        let start = crate::align(self.pos, T::BITS);

        // The head up to a row boundary and the tail past the last whole row are short, so
        // they go through the serial fill, which also moves the position.
        let head = ((1024 - (start & 1023)) / w) as usize % per_row;
        let head = head.min(out.len());
        let (head_out, rest) = out.split_at_mut(head);
        self.fill(head_out);
        let rows_len = rest.len() / per_row * per_row;
        let (rows, tail) = rest.split_at_mut(rows_len);

        let (key, k) = (self.key, self.k);
        let base = self.pos;
        rows.par_chunks_mut(TASK_ROWS * per_row)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut task = Tandem::from_key(key, base + 1024 * (i * TASK_ROWS) as u64, k);
                task.fill(chunk);
            });
        self.pos = base + w * rows.len() as u64;
        self.fill(tail);
    }

    /// [`fill_u32`](Self::fill_u32) across threads. The values and the final position are
    /// the serial fill's.
    pub fn par_fill_u32(&mut self, out: &mut [u32]) {
        self.par_fill(out)
    }
    /// [`fill_u64`](Self::fill_u64) across threads.
    pub fn par_fill_u64(&mut self, out: &mut [u64]) {
        self.par_fill(out)
    }
    /// [`fill_f32`](Self::fill_f32) across threads.
    pub fn par_fill_f32(&mut self, out: &mut [f32]) {
        self.par_fill(out)
    }
    /// [`fill_below_u32`](Self::fill_below_u32) across threads. Element `i` depends on draw `i`
    /// alone, so the fill and the bounding both split freely.
    pub fn par_fill_below_u32(&mut self, out: &mut [u32], n: u32) {
        self.par_fill_u32(out);
        let me = &*self;
        out.par_chunks_mut(TASK_ROWS * 32)
            .enumerate()
            .for_each(|(i, chunk)| me.bound_u32(chunk, (i * TASK_ROWS * 32) as u64, n));
    }
    /// [`fill_below_u64`](Self::fill_below_u64) across threads.
    pub fn par_fill_below_u64(&mut self, out: &mut [u64], n: u64) {
        self.par_fill_u64(out);
        let me = &*self;
        out.par_chunks_mut(TASK_ROWS * 16)
            .enumerate()
            .for_each(|(i, chunk)| me.bound_u64(chunk, (i * TASK_ROWS * 16) as u64, n));
    }
    /// [`fill_normal_f64`](Self::fill_normal_f64) across threads. Each task starts at its
    /// pair's draws, so no scratch buffer for the uniforms is needed.
    pub fn par_fill_normal_f64(&mut self, out: &mut [f64]) {
        self.par_normals(out, 64, Tandem::fill_normal_f64)
    }
    /// [`fill_normal_f32`](Self::fill_normal_f32) across threads.
    pub fn par_fill_normal_f32(&mut self, out: &mut [f32]) {
        self.par_normals(out, 32, Tandem::fill_normal_f32)
    }

    fn par_normals<T: Send>(
        &mut self,
        out: &mut [T],
        w: u32,
        fill: impl Fn(&mut Tandem, &mut [T]) + Sync,
    ) {
        // An even task size keeps every pair inside one task.
        const TASK: usize = 8192;
        let base = crate::align(self.pos, w);
        let (key, k) = (self.key, self.k);
        let w = u64::from(w);
        out.par_chunks_mut(TASK).enumerate().for_each(|(i, chunk)| {
            let mut task = Tandem::from_key(key, base + w * (i * TASK) as u64, k);
            fill(&mut task, chunk);
        });
        self.pos = base + w * out.len().next_multiple_of(2) as u64;
    }

    /// [`fill_f64`](Self::fill_f64) across threads.
    pub fn par_fill_f64(&mut self, out: &mut [f64]) {
        self.par_fill(out)
    }
}
