from distutils.core import setup

setup(
    install_requires=["alembic==1.9.1; python_version >= '3.7'", 'ansi2html==1.8.0', "anyio==3.6.2; python_full_version >= '3.6.2'", "argon2-cffi==21.3.0; python_version >= '3.6'", "argon2-cffi-bindings==21.2.0; python_version >= '3.6'", "arrow==1.2.3; python_version >= '3.6'", 'asttokens==2.2.1', "attrs==22.2.0; python_version >= '3.6'", "autopage==0.5.1; python_version >= '3.6'", 'backcall==0.2.0', "beautifulsoup4==4.11.1; python_full_version >= '3.6.0'", "bidict==0.22.1; python_version >= '3.7'", "bleach==5.0.1; python_version >= '3.7'", "certifi==2022.12.7; python_version >= '3.6'", 'cffi==1.15.1', "cfgv==3.3.1; python_full_version >= '3.6.1'", "charset-normalizer==2.1.1; python_full_version >= '3.6.0'", "click==8.1.3; python_version >= '3.7'", "cliff==4.1.0; python_version >= '3.8'", "cloudpickle==2.2.0; python_version >= '3.6'", "cmaes==0.9.1; python_version >= '3.7'", "cmd2==2.4.2; python_version >= '3.6'", "colorlog==6.7.0; python_version >= '3.6'", "comm==0.1.2; python_version >= '3.6'", "contourpy==1.0.6; python_version >= '3.7'", "cycler==0.11.0; python_version >= '3.6'", 'cython==0.29.33', 'databricks-cli==0.17.4', "debugpy==1.6.5; python_version >= '3.7'", "decorator==5.1.1; python_version >= '3.5'", "defusedxml==0.7.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'", 'dill==0.3.6', 'distlib==0.3.6', "docker==6.0.1; python_version >= '3.7'", 'ecos==2.0.12', "entrypoints==0.4; python_version >= '3.6'", "exceptiongroup==1.1.0; python_version < '3.11'", 'executing==1.2.0', 'fastjsonschema==2.16.2', "filelock==3.9.0; python_version >= '3.7'", "flask==2.2.2; python_version >= '3.7'", 'flask-socketio==5.3.2', "fonttools==4.38.0; python_version >= '3.7'", 'fqdn==1.5.1', 'frozendict==2.3.4', "gitdb==4.0.10; python_version >= '3.7'", "gitpython==3.1.30; python_version >= '3.7'", "greenlet==2.0.1; python_version >= '3' and (platform_machine == 'aarch64' or (platform_machine == 'ppc64le' or (platform_machine == 'x86_64' or (platform_machine == 'amd64' or (platform_machine == 'AMD64' or (platform_machine == 'win32' or platform_machine == 'WIN32'))))))", "gunicorn==20.1.0; platform_system != 'Windows'", "identify==2.5.12; python_version >= '3.7'", "idna==3.4; python_version >= '3.5'", 'imbalanced-learn==0.10.1', 'imblearn==0.0', "importlib-metadata==4.13.0; python_version >= '3.7'", "iniconfig==2.0.0; python_version >= '3.7'", "ipykernel==6.20.1; python_version >= '3.8'", "ipython==8.8.0; python_version >= '3.8'", 'ipython-genutils==0.2.0', 'isoduration==20.11.0', "itsdangerous==2.1.2; python_version >= '3.7'", "jedi==0.18.2; python_version >= '3.6'", "jinja2==3.1.2; platform_system != 'Windows'", "joblib==1.2.0; python_version >= '3.7'", 'jsonpointer==2.3', "jsonschema==4.17.3; python_version >= '3.7'", "jupyter-client==7.4.8; python_version >= '3.7'", "jupyter-core==5.1.3; python_version >= '3.8'", "jupyter-events==0.6.0; python_version >= '3.7'", "jupyter-server==2.0.6; python_version >= '3.8'", "jupyter-server-terminals==0.4.4; python_version >= '3.8'", "jupyterlab-pygments==0.2.2; python_version >= '3.7'", "kiwisolver==1.4.4; python_version >= '3.7'", "llvmlite==0.39.1; python_version >= '3.7'", "mako==1.2.4; python_version >= '3.7'", "markdown==3.4.1; python_version >= '3.7'", "markupsafe==2.1.1; python_version >= '3.7'", 'matplotlib==3.6.2', "matplotlib-inline==0.1.6; python_version >= '3.5'", 'matplotlib-label-lines==0.5.1', 'mistune==2.0.4', 'mlflow==2.1.1', "more-itertools==9.0.0; python_version >= '3.7'", 'multimethod==1.9.1', "nbclassic==0.4.8; python_version >= '3.7'", "nbclient==0.7.2; python_full_version >= '3.7.0'", "nbconvert==7.2.7; python_version >= '3.7'", "nbformat==5.7.1; python_version >= '3.7'", "nest-asyncio==1.5.6; python_version >= '3.5'", "nodeenv==1.7.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6'", 'notebook==6.5.2', "notebook-shim==0.2.2; python_version >= '3.7'", "numba==0.56.4; python_version >= '3.7'", "numexpr==2.8.4; python_version >= '3.7'", 'numpy==1.23.5', "oauthlib==3.2.2; python_version >= '3.6'", 'optuna==3.0.5', 'osqp==0.6.2.post8', "packaging==22.0; python_version >= '3.7'", 'pandas==1.5.2', "pandocfilters==1.5.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", "parso==0.8.3; python_version >= '3.6'", 'patsy==0.5.3', "pbr==5.11.0; python_version >= '2.6'", "pexpect==4.8.0; sys_platform != 'win32'", 'pickleshare==0.7.5', "pillow==9.4.0; python_version >= '3.7'", "platformdirs==2.6.2; python_version >= '3.7'", 'plotly==5.11.0', "pluggy==1.0.0; python_version >= '3.6'", 'pre-commit==2.21.0', "prettytable==3.6.0; python_version >= '3.7'", "prometheus-client==0.15.0; python_version >= '3.6'", "prompt-toolkit==3.0.36; python_full_version >= '3.6.2'", "protobuf==4.21.12; python_version >= '3.7'", "psutil==5.9.4; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", "ptyprocess==0.7.0; os_name != 'nt'", 'pure-eval==0.2.2', 'pyaml==21.10.1', "pyarrow==10.0.1; python_version >= '3.7'", 'pycparser==2.21', "pygments==2.14.0; python_version >= '3.6'", 'pyhumps==3.8.0', "pyjwt==2.6.0; python_version >= '3.7'", "pyparsing==3.0.9; python_full_version >= '3.6.8'", 'pyperclip==1.8.2', "pyrsistent==0.19.3; python_version >= '3.7'", 'pytest==7.2.0', 'pytest-mock==3.10.0', "python-dateutil==2.8.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2'", "python-engineio==4.3.4; python_version >= '3.6'", "python-json-logger==2.0.4; python_version >= '3.5'", "python-socketio==5.7.2; python_version >= '3.6'", 'pytz==2022.7', "pyyaml==6.0; python_version >= '3.6'", "pyzmq==24.0.1; python_version >= '3.6'", 'qdldl==0.1.5.post2', 'querystring-parser==1.2.4', "requests==2.28.1; python_version >= '3.7' and python_version < '4'", 'rfc3339-validator==0.1.4', 'rfc3986-validator==0.1.1', 'scikit-learn==1.1.3', 'scikit-survival==0.19.0.post1', "scipy==1.8.1; python_version < '3.11' and python_version >= '3.8'", 'send2trash==1.8.0', "setuptools==65.6.3; python_version >= '3.7'", 'shap==0.41.0', "six==1.16.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2'", "slicer==0.0.7; python_version >= '3.6'", "smmap==5.0.0; python_version >= '3.6'", "sniffio==1.3.0; python_version >= '3.7'", "soupsieve==2.3.2.post1; python_version >= '3.6'", "sqlalchemy==1.4.46; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'", "sqlparse==0.4.3; python_version >= '3.5'", 'stack-data==0.6.2', 'statsmodels==0.13.5', "stevedore==4.1.1; python_version >= '3.8'", "tabulate==0.9.0; python_version >= '3.7'", "tenacity==8.1.0; python_version >= '3.6'", "terminado==0.17.1; python_version >= '3.7'", "threadpoolctl==3.1.0; python_version >= '3.6'", "tinycss2==1.2.1; python_version >= '3.7'", "tomli==2.0.1; python_version < '3.11'", 'toolz==0.12.0', 'torchtuples==0.2.2', "tornado==6.2; python_version >= '3.7'", "tqdm==4.64.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", "traitlets==5.8.1; python_version >= '3.7'", 'uri-template==1.2.0', "urllib3==1.26.13; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'", "virtualenv==20.17.1; python_version >= '3.6'", 'wcwidth==0.2.5', 'webcolors==1.12', 'webencodings==0.5.1', "websocket-client==1.4.2; python_version >= '3.7'", "werkzeug==2.2.2; python_version >= '3.7'", 'xgboost==1.7.3', "zipp==3.11.0; python_version >= '3.7'"































































































],
    name="hcve_lib",
    version="0.1.10",
    packages=[""],
    url="",
    license="",
    author="sitnarf",
    author_email="sitnarf@gmail.com",
    description="HCVE ML commons",
)