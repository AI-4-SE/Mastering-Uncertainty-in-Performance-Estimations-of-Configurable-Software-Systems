import setuptools

setuptools.setup(
    name="activesampler",  # Replace with your own username
    version="0.0.1",
    author="Johannes Dorn",
    author_email="johannes.dorn@uni-leipzig.de",
    description="Auxillary tools to process performance samples",
    long_description_content_type="text/markdown",
    url="https://github.com/AI-4-SE/Mastering-Uncertainty-in-Performance-Estimations-of-Configurable-Software-Systems",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'matplotlib>=3.2.1,<3.3.3',
        'pandas>=1.0.1,<1.0.5',
        'networkx>=2.4,<2.5',
        'statsmodels>=0.11.1,<0.12'
    ],
)
