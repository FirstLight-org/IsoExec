"""Bind a project-x GLM projection manifest to its SM100 implementation."""
from dataclasses import dataclass
import hashlib
import json


# SHA-256 of the compiler's emitted glm.checkpoint program. K and N are
# parameters of the program. Its quantizer, reducer and merge tree are bound.
_SOURCE_BY_MAX_TP = {
    1: "892633aa61851e2bc8990ac5c0b23a0ccb0f30594cdcf6713ec25947d0dfc15c",
    2: "e0fdcaf86793cc435bd6ae9db6b9d4241cca91863ab650208fca5f0cc982117c",
    4: "a66b60f869aac863beb8b74e641c46a84d4175997fe91300693eeb90a039b91f",
    8: "0aa7ab61ce02b8fc839816ca79245c2590adc08b219b965ea96e1a07d67b4d9a",
}


@dataclass(frozen=True)
class ProjectionSchedule:
    k: int
    n: int
    segment_length: int
    max_tp: int
    manifest_hash: str

    @classmethod
    def from_manifest(cls, manifest):
        reduction = manifest["reductions"]["y"]
        k, n = manifest["shape"]["K"], manifest["shape"]["N"]
        degree = reduction["d_max"]
        if (manifest["region"] != "glm_checkpoint_fp8" or manifest["table"] != "sm100.lowbit"
                or manifest["devices"] != ["sm100"] or degree not in _SOURCE_BY_MAX_TP
                or manifest["source_sha256"] != _SOURCE_BY_MAX_TP[degree]
                or reduction["reducer"] != "scalefma128.e4m3_e4m3" or reduction["block"] != 128
                or reduction["merge"] != "add32" or reduction["S"] != "one per shard"):
            raise ValueError("unsupported GLM FP8 numerical schedule")
        length = reduction["n"]
        if n < 1 or length < 128 or length % 128 or k != degree * length:
            raise ValueError("invalid FP8 segment length for this weight")
        encoded = json.dumps({k: v for k, v in manifest.items() if k != "notes"}, sort_keys=True)
        digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        return cls(k, n, length, degree, digest)

    def local_segments(self, n, k, tp_degree):
        if (type(tp_degree) is not int or tp_degree < 1 or self.max_tp % tp_degree
                or n != self.n or k * tp_degree != self.k):
            raise ValueError("local projection dimensions or TP degree disagree with the policy manifest")
        return k // self.segment_length
