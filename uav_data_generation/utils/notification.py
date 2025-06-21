import functools
import smtplib
import time
from email.message import EmailMessage
from typing import Callable, Optional

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


def _load_env(env_file: str, usecwd: bool = False):
    env_file = dotenv.find_dotenv(filename=env_file, usecwd=usecwd)
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
        logger.info(":white_check_mark: The notification is sended!")
    except Exception as e:
        logger.exception(e)


def notify(
    env_file: str = ".env", task_name: Optional[str] = None, usecwd: bool = False
) -> Callable:
    def decorator(func: Callable) -> Callable:
        env_vars = _load_env(env_file, usecwd=usecwd)
        if env_vars is None:
            return func
        if not bool(int(env_vars["ENABLE_NOTIFICATION"])):
            return func

        conn = setup_email(env_vars)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # get logger during actual execution to enter the correct context
            logger = logging.get_logger()
            logger.info(":white_check_mark: Notification is enabled!")

            is_failed = False
            begin = time.time()
            try:
                return func(*args, **kwargs)
            except Exception:
                is_failed = True
                raise
            finally:
                delta = time.gmtime(time.time() - begin)
                ftime = time.strftime("%H:%M:%S", delta)

                # prepare content
                state = "unsuccessfully" if is_failed else "successfully"
                content = f"is executed {state} after {ftime}."
                nonlocal task_name
                if task_name is None:
                    task_name = f"{func.__name__}() in {__file__}"
                content = f"{task_name} {content}"

                logger.info(f":white_check_mark: {content}")
                send_email(conn, content, env_vars)
                conn.quit()

        return wrapper

    return decorator
