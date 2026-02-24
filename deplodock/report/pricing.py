"""GPU pricing helpers."""


def get_gpu_price(config: dict, gpu_type: str, gpu_count: int) -> float:
    """Get GPU price from config."""
    if 'pricing' not in config:
        return 0.0

    pricing = config['pricing']
    gpu_type_normalized = gpu_type.lower()

    if gpu_type_normalized in pricing:
        price_per_gpu = pricing[gpu_type_normalized]
        return price_per_gpu * gpu_count

    variations = {
        'rtx4090': ['4090', 'rtx_4090'],
        'rtx5090': ['5090', 'rtx_5090'],
        'pro6000': ['6000', 'rtx_6000', 'rtx6000', 'quadro_rtx_6000']
    }

    for base_name, alternatives in variations.items():
        if gpu_type_normalized in alternatives or base_name == gpu_type_normalized:
            if base_name in pricing:
                price_per_gpu = pricing[base_name]
                return price_per_gpu * gpu_count

    return 0.0
