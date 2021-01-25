
from setuptools import setup, find_packages

with open("./README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="glasses", 
    version="0.0.6",
    author="Francesco Saverio Zuppichini & Francesco Cicala",
    author_email="francesco.zuppichini@gmail.com",
    description="Compact, concise and customizable deep learning computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrancescoSaverioZuppichini/glasses",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
