import functools
import smtplib
import time
from email.message import EmailMessage
from typing import Callable

import dotenv

from .. import logging


class EmailSettings:
    SMTP_HOST = "SMTP_HOST"
    SMTP_PORT = "SMTP_PORT"
    SMTP_USERNAME = "SMTP_USERNAME"
    SMTP_PASSWORD = "SMTP_PASSWORD"
    FROM = "EMAIL_FROM"
    TO = "EMAIL_TO"
    TITLE = "EMAIL_TITLE"


def _load_env(env_file: str):
    env_file = dotenv.find_dotenv(filename=env_file)
    env_vars = dotenv.dotenv_values(dotenv_path=env_file)
    return env_vars if len(env_vars) != 0 else None


def setup_email(env_vars: dict[str, str]):
    smtp_host = env_vars[EmailSettings.SMTP_HOST]
    smtp_port = int(env_vars[EmailSettings.SMTP_PORT])
    smtp_username = env_vars[EmailSettings.SMTP_USERNAME]
    smtp_password = env_vars[EmailSettings.SMTP_PASSWORD]

    smtp = smtplib.SMTP(host=smtp_host, port=smtp_port)
    smtp.starttls()
    smtp.login(smtp_username, smtp_password)
    return smtp


def send_email(conn: smtplib.SMTP, content: str, env_vars: dict[str, str]):
    sender = env_vars[EmailSettings.FROM]
    receivers = env_vars[EmailSettings.TO].split(",")
    title = env_vars[EmailSettings.TITLE]

    msg = EmailMessage()
    msg["Subject"] = title
    msg.set_content(content)

    logger = logging.get_logger()
    try:
        conn.send_message(msg, from_addr=sender, to_addrs=receivers)
        logger.info("[bold bright_green] Sended the notification successfully!...")
    except Exception as e:
        logger.exception(e)


def notify(env_file: str = ".env") -> Callable:
    def decorator(func: Callable) -> Callable:
        env_vars = _load_env(env_file)
        if env_vars is None:
            return func
        if not bool(env_vars["ENABLE_NOTIFICATION"]):
            return func

        conn = setup_email(env_vars)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # get logger during actual execution to enter the correct context
            logger = logging.get_logger()
            logger.info(":white_check_mark: Notification is enabled!")

            begin = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                ftime = time.strftime("%d:%H:%M:%S", time.gmtime(time.time() - begin))
                content = f"{func.__name__} is finished after {ftime}."

                logger.info(f":white_check_mark: {content}")
                send_email(conn, content, env_vars)
                conn.quit()

        return wrapper

    return decorator
