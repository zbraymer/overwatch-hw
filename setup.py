from setuptools import setup
from _version import __version__

setup(
    name='rail_finder',
    version=__version__,
    description='This tool accepts an image and attempts to fine the centerline'
                ' of railroad tracks. It should return the same image, renamed,'
                ' with a red line for the tracks centerline',
    author='Zach Raymer',
    author_email='rayme1zb@gmail.com',
    license='MIT',
    entry_points={
        'console_scripts': [
            'rail_finder = rail_finder.rail_finder:main',
        ],
    }
)
