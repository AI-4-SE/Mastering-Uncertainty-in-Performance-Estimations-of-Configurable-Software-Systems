import setuptools

setuptools.setup(
    name="p4",  # Replace with your own username
    version="0.0.1",
    author="Johannes Dorn",
    author_email="johannes.dorn@uni-leipzig.de",
    description="Performance Predictions with Probabilistic Programming",
    long_description_content_type="text/markdown",
    url="https://github.com/AI-4-SE/Mastering-Uncertainty-in-Performance-Estimations-of-Configurable-Software-Systems",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'matplotlib>=3.2.1,<3.3.3',
        'pandas>=1.0.1,<1.0.5',
        'arviz>=0.7.0,<0.8.0',
        'seaborn>=0.9.1,<0.11',
        'scipy>=1.4.1,<1.5',
        'scikit-learn>=0.22,<0.23',
        'scipy>=1.4.1,<1.5',
    ],
)
