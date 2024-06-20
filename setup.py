from distutils.core import setup

import setuptools

setup(
    dependency_links=[],
    install_requires=["alembic==1.12.0; python_version >= '3.7'", 'ansi2html==1.8.0', "anyio==4.0.0; python_version >= '3.8'", "argon2-cffi==23.1.0; python_version >= '3.7'", "argon2-cffi-bindings==21.2.0; python_version >= '3.6'", "arrow==1.2.3; python_version >= '3.6'", 'asttokens==2.4.0', "async-lru==2.0.4; python_version >= '3.8'", "attrs==23.1.0; python_version >= '3.7'", "babel==2.12.1; python_version >= '3.7'", 'backcall==0.2.0', "beautifulsoup4==4.12.2; python_full_version >= '3.6.0'", "bidict==0.22.1; python_version >= '3.7'", "bleach==6.0.0; python_version >= '3.7'", "blinker==1.6.2; python_version >= '3.7'", "certifi==2023.7.22; python_version >= '3.6'", 'cffi==1.15.1', "cfgv==3.4.0; python_version >= '3.8'", "charset-normalizer==3.2.0; python_full_version >= '3.7.0'", "click==8.1.7; python_version >= '3.7'", "cloudpickle==2.2.1; python_version >= '3.6'", "cmaes==0.10.0; python_version >= '3.7'", "colorama==0.4.6; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6'", "colorlog==6.7.0; python_version >= '3.6'", "comm==0.1.4; python_version >= '3.6'", "contourpy==1.1.0; python_version >= '3.8'", "cycler==0.11.0; python_version >= '3.6'", 'cython==3.0.2', 'databricks-cli==0.17.7', "debugpy==1.6.7.post1; python_version >= '3.7'", "decorator==5.1.1; python_version >= '3.5'", "defusedxml==0.7.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'", 'dill==0.3.7', 'distlib==0.3.7', "docker==6.1.3; python_version >= '3.7'", "entrypoints==0.4; python_version >= '3.6'", 'executing==1.2.0', 'fastjsonschema==2.18.0', "filelock==3.12.3; python_version >= '3.8'", "flask==2.3.3; python_version >= '3.8'", 'flask-socketio==5.3.6', "fonttools==4.42.1; python_version >= '3.8'", 'fqdn==1.5.1', 'frozendict==2.3.8', "gitdb==4.0.10; python_version >= '3.7'", "gitpython==3.1.34; python_version >= '3.7'", "greenlet==2.0.2; platform_machine == 'aarch64' or (platform_machine == 'ppc64le' or (platform_machine == 'x86_64' or (platform_machine == 'amd64' or (platform_machine == 'AMD64' or (platform_machine == 'win32' or platform_machine == 'WIN32')))))", "identify==2.5.27; python_version >= '3.8'", "idna==3.4; python_version >= '3.5'", 'imbalanced-learn==0.11.0', 'imblearn==0.0', "importlib-metadata==6.8.0; python_version >= '3.8'", "iniconfig==2.0.0; python_version >= '3.7'", "ipykernel==6.25.2; python_version >= '3.8'", "ipython==8.15.0; python_version >= '3.9'", 'isoduration==20.11.0', "itsdangerous==2.1.2; python_version >= '3.7'", "jedi==0.19.0; python_version >= '3.6'", "jinja2==3.1.2; platform_system == 'Windows'", "joblib==1.3.2; python_version >= '3.7'", 'json5==0.9.14', 'jsonpointer==2.4', "jsonschema==4.19.0; python_version >= '3.8'", "jsonschema-specifications==2023.7.1; python_version >= '3.8'", "jupyter-client==8.3.1; python_version >= '3.8'", "jupyter-core==5.3.1; python_version >= '3.8'", "jupyter-events==0.7.0; python_version >= '3.8'", "jupyter-lsp==2.2.0; python_version >= '3.8'", "jupyter-server==2.7.3; python_version >= '3.8'", "jupyter-server-terminals==0.4.4; python_version >= '3.8'", "jupyterlab==4.0.5; python_version >= '3.8'", "jupyterlab-pygments==0.2.2; python_version >= '3.7'", "jupyterlab-server==2.24.0; python_version >= '3.7'", "kiwisolver==1.4.5; python_version >= '3.7'", "mako==1.2.4; python_version >= '3.7'", "markdown==3.4.4; python_version >= '3.7'", 'markupsafe==2.1.3', 'matplotlib==3.7.2', "matplotlib-inline==0.1.6; python_version >= '3.5'", 'matplotlib-label-lines==0.6.0', "mistune==3.0.1; python_version >= '3.7'", 'mlflow==2.6.0', "more-itertools==10.1.0; python_version >= '3.8'", 'multimethod==1.9.1', "nbclient==0.8.0; python_full_version >= '3.8.0'", "nbconvert==7.8.0; python_version >= '3.8'", "nbformat==5.9.2; python_version >= '3.8'", "nest-asyncio==1.5.7; python_version >= '3.5'", "nodeenv==1.8.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6'", 'notebook==7.0.3', "notebook-shim==0.2.3; python_version >= '3.7'", 'numpy==1.25.2', "oauthlib==3.2.2; python_version >= '3.6'", 'optuna==3.3.0', "overrides==7.4.0; python_version >= '3.6'", "packaging==23.1; python_version >= '3.7'", 'pandas==2.1.0', "pandocfilters==1.5.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", "parso==0.8.3; python_version >= '3.6'", 'patsy==0.5.3', 'pickleshare==0.7.5', "pillow==10.0.0; python_version >= '3.8'", "platformdirs==3.10.0; python_version >= '3.7'", 'plotly==5.16.1', "pluggy==1.3.0; python_version >= '3.8'", 'pre-commit==3.4.0', "prometheus-client==0.17.1; python_version >= '3.6'", "prompt-toolkit==3.0.39; python_full_version >= '3.7.0'", "protobuf==4.24.2; python_version >= '3.7'", "psutil==5.9.5; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", 'pure-eval==0.2.2', 'pyaml==23.9.1', "pyarrow==12.0.1; python_version >= '3.7'", 'pycparser==2.21', "pygments==2.16.1; python_version >= '3.7'", 'pyhumps==3.8.0', "pyjwt==2.8.0; python_version >= '3.7'", "pyparsing==3.0.9; python_full_version >= '3.6.8'", 'pytest==7.4.1', 'pytest-mock==3.11.1', "python-dateutil==2.8.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2'", "python-engineio==4.7.0; python_version >= '3.6'", "python-json-logger==2.0.7; python_version >= '3.6'", "python-socketio==5.9.0; python_version >= '3.6'", 'pytz==2023.3.post1', "pywin32==306; sys_platform == 'win32'", "pywinpty==2.0.11; os_name == 'nt'", "pyyaml==6.0.1; python_version >= '3.6'", "pyzmq==25.1.1; python_version >= '3.6'", 'querystring-parser==1.2.4', "referencing==0.30.2; python_version >= '3.8'", "requests==2.31.0; python_version >= '3.7'", "rfc3339-validator==0.1.4; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'", "rfc3986-validator==0.1.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'", "rpds-py==0.10.2; python_version >= '3.8'", 'scikit-learn==1.3.0', "scipy==1.11.2; python_version < '3.13' and python_version >= '3.9'", "send2trash==1.8.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'", "setuptools==68.1.2; python_version >= '3.8'", "six==1.16.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2'", "smmap==5.0.0; python_version >= '3.6'", "sniffio==1.3.0; python_version >= '3.7'", "soupsieve==2.5; python_version >= '3.8'", "sqlalchemy==2.0.20; python_version >= '3.7'", "sqlparse==0.4.4; python_version >= '3.5'", 'stack-data==0.6.2', 'statsmodels==0.14.0', "tabulate==0.9.0; python_version >= '3.7'", "tenacity==8.2.3; python_version >= '3.7'", "terminado==0.17.1; python_version >= '3.7'", "threadpoolctl==3.2.0; python_version >= '3.8'", "tinycss2==1.2.1; python_version >= '3.7'", 'toolz==0.12.0', 'torchtuples==0.2.2', "tornado==6.3.3; python_version >= '3.8'", "tqdm==4.66.1; python_version >= '3.7'", "traitlets==5.9.0; python_version >= '3.7'", "typing-extensions==4.7.1; python_version >= '3.7'", "tzdata==2023.3; python_version >= '2'", "tzlocal==5.0.1; python_version >= '3.7'", 'uri-template==1.3.0', "urllib3==1.26.16; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'", "virtualenv==20.24.4; python_version >= '3.7'", 'vistir==0.6.1', "waitress==2.1.2; platform_system == 'Windows'", 'wcwidth==0.2.6', 'webcolors==1.13', 'webencodings==0.5.1', "websocket-client==1.6.2; python_version >= '3.8'", "werkzeug==2.3.7; python_version >= '3.8'", 'xgboost==1.7.6', "zipp==3.16.2; python_version >= '3.8'"











































































],
    name="hcve_lib",
    version="0.1.13",
    packages=setuptools.find_packages(),
    url="",
    license="",
    author="sitnarf",
    author_email="sitnarf@gmail.com",
    description="HCVE ML commons",
)