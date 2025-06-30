from uav_data_generation.distributed import get_local_rank, get_rank, init_dist
from uav_data_generation.logging import get_logger

if __name__ == "__main__":
    logger = get_logger()
    init_dist()

    rank = get_rank()
    local_rank = get_local_rank()

    logger.info(f"Rank: {rank}, Local Rank: {local_rank}")
