import setuptools

with open("README.md") as readme:
    long_description = readme.read()

setuptools.setup(
    name="utils_hiv", 
    version="0.0.1",
    author="Luc Blassel",
    author_email="luc.blassel@gmail.com",
    description="small functions to study DRMs in HIV",
    long_description=long_description,
    packages = setuptools.find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'biopython',
        'statsmodels',
        'category_encoders',
        'scipy',
        'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True
)