import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paxplot",
    version="0.1.4b0",
    author="Jacob Kravits",
    author_email="kravitsjacob@gmail.com",
    description="Paxplot is a Python visualization library for parallel coordinate plots based on Matplotlib.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kravitsjacob/paxplot",
    project_urls={
        "Bug Tracker": "https://github.com/kravitsjacob/paxplot/issues",
    },
    install_requires=[
        'matplotlib>=3.5',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    include_package_data=True,
    package_data={'': ['data/tradeoff.csv', 'data/hydroclimate_model_evaluation.csv']},
)
