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
# soat_backend/README.md: contiguous 16x16, shapes multiples of 256; a temporary kernel, tunes its own configs
SOAT = KernelSpec("soat", "libpearl_soat.so", "16x16", 2048, 4096, 16384, 4096, (), None)

REGISTRY = {"7.0": V100, "8.0": SOAT, "8.6": SOAT, "8.9": SOAT, "12.0": SOAT}


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
