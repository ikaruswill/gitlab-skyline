# gitlab-skyline <!-- omit from toc -->
Generate a 3D Skyline in OpenSCAD and STL from GitLab contributions

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/gitlab-skyline.svg)](https://pypi.org/project/gitlab-skyline)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gitlab-skyline.svg)](https://pypi.org/project/gitlab-skyline) -->

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
  - [Quickstart](#quickstart)
  - [Arguments available](#arguments-available)
- [Features](#features)
  - [Matches GitLab contribution heatmap](#matches-gitlab-contribution-heatmap)
  - [Dynamic sizing for the current year](#dynamic-sizing-for-the-current-year)
  - [Customizable logo](#customizable-logo)
  - [Private GitLab installations](#private-gitlab-installations)
  - [Non-public user contribution history](#non-public-user-contribution-history)
  - [STL export](#stl-export)
- [License](#license)
- [Credits](#credits)

# Installation
1. Clone this repository
```
git clone https://github.com/ikaruswill/gitlab-skyline.git
```
2. Create and activate a virtual environment
```
cd gitlab-skyline && python -m venv env && source env/bin/activate
```
3. Install this package as an 'editable' install
```
pip install -e .
```
# Usage

## Quickstart
```
gitlab-skyline <username> <year>
```

## Arguments available
```
gitlab-skyline --help
usage: gitlab-skyline [-h] [-o OUTPUT] [--stl] [--domain DOMAIN] [--token TOKEN] [--concurrency CONCURRENCY] [--logo LOGO] [--loglevel LOGLEVEL]
                      username [year]

Create OpenSCAD [and STL] models from GitLab contributions

positional arguments:
  username              GitLab username (without @)
  year                  Year of contributions to fetch (default: 2023)

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output path (default: Current working directory)
  --stl                 Export an STL file as well (Requires openscad binary) (default: False)
  --domain DOMAIN       GitLab custom domain (default: https://gitlab.com)
  --token TOKEN         Personal access token (default: None)
  --concurrency CONCURRENCY
                        Max concurrent requests to GitLab (default: 2)
  --logo LOGO           Path to logo to be engraved on the front face (default: <script_path>/gitlab.svg)
  --loglevel LOGLEVEL   Log level (default: INFO)
```

# Features
## Matches GitLab contribution heatmap
The skyline model generated, when viewed from above, is identical to the contribution heatmap in the user's profile. Contributions on Sunday correspond to the row at the back and Saturday, to the row at the front, with the bars inbetween corresponding to the contributions on weekdays in order.

>Previous implementations generated models with contribution bars that did not match the profile heatmap.
## Dynamic sizing for the current year
If you left the `year` argument blank or set the year to be the current year, the length of the model will be dynamically adjusted based on the number of columns of contribution bars up till the current week. This is because dates that fall in the future are skipped and do not generate contribution bars.
## Customizable logo
By default, the GitLab logo is engraved on the left of the front face of the base.
 To provide a custom logo, supply the path to the SVG file in the `--logo` argument.
## Private GitLab installations
If your GitLab is self-hosted on a custom domain, you may specify the domain using the `--domain` argument.
## Non-public user contribution history
If you are not getting any contributions for the target user despite the heatmap being visible in the user's profile page, the user most likely has a private contribution history that is hidden from the public API.

To access private contribution history, [generate a Personal Access Token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html#create-a-personal-access-token) with the `read_user` scope, and supply it with the `--token` argument. 
## STL export
To enable STL export alongside SCAD export, supply the `--stl` argument. 

> Note: the `openscad` binary must be present on your machine.

Install OpenSCAD from https://www.openscad.org/downloads.html and ensure that the `openscad` binary is present and functional.
```
openscad --version
```
# License

`gitlab-skyline` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

# Credits
Based on the excellent work of [@felixgomez](https://github.com/felixgomez/gitlab-skyline)