from distutils.core import setup

setup(
    install_requires=[
        "alembic==1.6.5",
        "attrs==21.2.0",
        "backports.entry-points-selectable==1.1.0",
        "cfgv==3.3.0",
        "cliff==3.8.0",
        "cmaes==0.8.2",
        "cmd2==2.1.2",
        "colorama==0.4.4",
        "colorlog==5.0.1",
        "cycler==0.10.0",
        "distlib==0.3.2",
        "filelock==3.0.12",
        "frozendict==2.0.3",
        "greenlet==1.1.1; python_version >= '3'",
        "identify==2.2.13",
        "iniconfig==1.1.1",
        "joblib==1.0.1",
        "kiwisolver==1.3.1",
        "mako==1.1.4",
        "markupsafe==2.0.1",
        "matplotlib==3.4.2",
        "nodeenv==1.6.0",
        "numpy==1.21.1",
        "optuna==2.9.1",
        "packaging==21.0",
        "pandas==1.2",
        "pbr==5.6.0",
        "pillow==8.3.1",
        "platformdirs==2.2.0",
        "pluggy==0.13.1",
        "pre-commit==2.14.0",
        "prettytable==2.1.0",
        "py==1.10.0",
        "pyhumps==3.0.2",
        "pyparsing==2.4.7",
        "pyperclip==1.8.2",
        "pytest==6.2.4",
        "python-dateutil==2.8.2",
        "python-editor==1.0.4",
        "pytz==2021.1",
        "pyyaml==5.4.1",
        "scikit-learn==0.24.2",
        "scipy==1.7.1",
        "six==1.16.0",
        "sqlalchemy==1.4.22",
        "stevedore==3.3.0",
        "threadpoolctl==2.2.0",
        "toml==0.10.2",
        "toolz==0.11.1",
        "tqdm==4.62.0",
        "virtualenv==20.7.0",
        "wcwidth==0.2.5",
        "xgboost==1.4.2",
    ],
    name="hcve_lib",
    version="0.1",
    packages=[""],
    url="",
    license="",
    author="sitnarf",
    author_email="sitnarf@gmail.com",
    description="HCVE ML commons",
)
