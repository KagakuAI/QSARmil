from setuptools import setup, find_packages

setup(
    name="qsarmil",
    version="1.0",
    author="KagakuAI",
    author_email="dvzankov@gmail.com.com",
    description="Molecular multi-instance machine learning",
    long_description_content_type="text/x-rst",
    url="https://github.com/KagakuAI/QSARmil",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "matplotlib",
        "torch",
        "torchvision",
        "torch_optimizer",
        "rdkit",
        "molfeat",
        "tqdm"

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # update if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
