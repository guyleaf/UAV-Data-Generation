from uav_data_generation import logging
from uav_data_generation.utils.notification import notify


@notify(task_name="test task")
def test():
    logger = logging.get_logger()
    logger.info("Test notification~~~")


if __name__ == "__main__":
    test()
