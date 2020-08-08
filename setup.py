import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cfed",
    version="0.0.1",
    author="Wajdy",
    author_email="itswajdy@gmail.com",
    description="A python package to compute pairwise Euclidean distances on datasets with categorical features in little time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ItsWajdy/categorical_features_euclidean_distance",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)