"""Kernel registry: the compute capability of a card (nvidia-smi compute_cap) -> the kernel library and the work shape
it mines with. A card without an entry, or whose library is missing from --kernels-dir, is not mined (NoKernel)."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelSpec:
    name: str                   # "kernel" of /summary
    lib: str                    # file name in --kernels-dir
    tile: str                   # jobs.TILES key the library is built for
    k: int
    m: int
    n: int
    portion_rows: int
    variants: tuple[str, ...]   # PEARL_VARIANT values to tune over when the library does not export pearl_variants
    k_passport: float | None    # expected MAC/(SM*clock) until the card's own tune measures it


# kernels/v100/README.md: tile v100, k = n = 2048 (B'^T stays in L2), m 65 536 in one portion; three CTA shapes;
# k 331 -- v1 `search` on 8x V100 at 21920 (TZ_v100_kernel_v2.md §1)
V100 = KernelSpec("v100", "libpearl_v100.so", "v100", 2048, 65536, 2048, 65536,
                  ("v100-hmma884-128x256", "v100-hmma884-256x128", "v100-hmma884-128x128"), 331.0)
# kernels/sm80/README.md: tile 8x16, one fat binary for sm_80/86/89/120, operands built on the card (pearl_job /
# pearl_pass); the library exports its variants. n by L2: Ampere GA10x 3-6 MB keeps B'^T at n = k = 2048 (4 MB);
# A100 (40 MB), Ada and Blackwell (32-96 MB) take n = 8192 (16 MB) -- a pass 4x longer, so building A' (m*k bytes per
# pass) costs ~4% instead of ~15% (kernels/common/README.md).
SM80_SMALL_L2 = KernelSpec("sm80", "libpearl_sm80.so", "8x16", 2048, 65536, 2048, 65536, (), None)
SM80 = KernelSpec("sm80", "libpearl_sm80.so", "8x16", 2048, 65536, 8192, 65536, (), None)

REGISTRY = {"7.0": V100, "8.0": SM80, "8.6": SM80_SMALL_L2, "8.9": SM80, "12.0": SM80}


class NoKernel(LookupError):
    pass


def kernel_for(compute_cap: str | None, kernels_dir: Path) -> tuple[KernelSpec, Path]:
    """(spec, library path) of a card; NoKernel when the registry has none or the library is not in kernels_dir."""
    spec = REGISTRY.get(compute_cap)
    if spec is None:
        raise NoKernel(f"no kernel for compute capability {compute_cap} (registry: {', '.join(REGISTRY)})")
    path = Path(kernels_dir).resolve() / spec.lib
    if not path.is_file():
        raise NoKernel(f"kernel {spec.name}: {path} not found")
    return spec, path
