// SPDX-License-Identifier: Apache-2.0
// Engine-record profile (record.cuh). Both profiles are the same recipe and bit for bit the same outputs; they
// differ in which cached quantities of the open chunk the record stores. record_layout.py reads this file.
//
// KDA_REC_RECOMPUTE_KI: do not store ki^b (12,288 bytes a head); every decode step of block b >= 1 recomputes
// it from the stored k^ and G rows by the block-entry code. Saves 6.5% of the record for a measured decode
// cost; off by default.
// #define KDA_REC_RECOMPUTE_KI 1
#pragma once
