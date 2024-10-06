import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()

    return args


def cli():
    args = parse_args()
    print(args)


if __name__ == "__main__":
    cli()
