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
        'pandas==0.25.3',
        'scikit-learn==0.20.3',
        'biopython==1.74',
        'statsmodels==0.9.0',
        'category_encoders==1.3.0',
        'scipy==1.10.0',
        'numpy==1.22.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True
)
