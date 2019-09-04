import os
import logging
import click
from pathlib import Path


def get_project_path():
    try:
        # running from a python file"
        project_path = Path(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        is_file = True
    except NameError as e:
        # not running from a file (ie console or notebook)
        project_path = Path(os.getcwd(), os.pardir)
        is_file = False
    return project_path, is_file

def conditionally_decorate(decorator, condition):
    '''return decorator that applies a passed decorator if some condition is True

    See https://stackoverflow.com/questions/3773555/python3-decorating-conditionally
    '''
    def resdec(f):
        if not condition:
            return f
        return decorator(f)
    return resdec


project_path, is_file = get_project_path()

@conditionally_decorate(
    click.command(), 
    is_file
)
@conditionally_decorate(
    click.option('--normalize', default=None, help='how to normalize data, possible values: {"standard", "minimax"}.'), 
    is_file
)
@conditionally_decorate(
    click.option('--nr_images', default=81, help='Number of images to generate for training'), 
    is_file
)
def test(normalize="none", nr_images=83):
    print("test")
    print(f"normalize: {normalize}")
    print(f"nr_images: {nr_images}")

if __name__ == '__main__':
        test()