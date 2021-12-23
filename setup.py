import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paxplot",
    version="0.0.14",
    author="Jacob Kravits",
    author_email="kravitsjacob@gmail.com",
    description="Create static parallel axis plots in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kravitsjacob/paxplot",
    project_urls={
        "Bug Tracker": "https://github.com/kravitsjacob/paxplot/issues",
    },
    install_requires=[
        'numpy>=1.16',
        'matplotlib>=3.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)