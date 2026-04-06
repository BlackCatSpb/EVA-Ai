from setuptools import setup, find_packages

setup(
    name="eva",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "networkx",
        "nltk",
        "transformers",
        "torch",
        "pillow",
        "requests",
        "psutil",
        "nest-simulator"
    ],
)