<div align="center">

 <img src="docs/resources/iEDA.png" width="15%" height="15%" alt="iEDA-logo" />
 <h1>iEDA</h1>
 <h3>An open-source RTL2GDS EDA platform for ASIC design.</h3>

[![License][License-img]][License-url]

**English** | [简体中文][README-CN-path]

</div>

Motivation ...

Design Philosophy ...

To get more information, please reference the iEDA homepage: [ieda.oscc.cc](https://ieda.oscc.cc/).

---

🎉 **Latest News:**

Check out the presentation: [**iEDA: An Open-Source Intelligent Physical lmplementation Toolkit and Library**][ISEDA-2023-iEDA-url] \[[paper][iEDA-paper], [slides][iEDA-slides]\] in Session 2 of ISEDA-2023 (9 May 2023, Nanjing,China), and discussion in Panel 6: [**Open-source EDA and Standards**][ISEDA-2023-panel6-url]

---

# Getting Started with iEDA

reference to our docs

## Build iEDA

### Install Dependencies

Ubuntu (recommend 20.04):
```bash
sudo bash build.sh -i apt
```

docker image:
```bash
docker pull iedaopensource/ieda-dev:0813
```

### build iEDA from source code

```bash
git clone iEDA_git_repo_address iEDA
cd iEDA
bash build.sh

# or build from docker:
docker run --rm -v $(pwd):/iEDA iedaopensource/ieda-dev:0813 ./iEDA/build.sh
```

### get iEDA binary release

docker package

## Try out iEDA

user manual

### iFlow

see iFlow

### Script

```shell
run fp
run pl
run cts
run rt
...
```

# Future Plan

roadmap

# How to Contribute

pull request (sign DCO)

# Discussion and Feedback

new issue

wechat group

# License

[Mulan Permissive Software License, Version 2 (Mulan PSL v2)][License-url]

<!-- links -->
[License-icon]: https://s2.d2scdn.com/static/imgs/favicon.ico
[License-img]: https://img.shields.io/badge/license-Mulan%20PSL%20v2-blue
[License-url]: LICENSE
[README-path]: README.md
[README-CN-path]: README.zh-cn.md
[ISEDA-2023-iEDA-url]: https://www.eda2.com/conferenceHome/program/detail?key=s2
[ISEDA-2023-panel6-url]: https://www.eda2.com/conferenceHome/program/detail?key=panel6
[iEDA-paper]: https://www.eda2.com/conferenceHome/program/detail?key=s2
[iEDA-slides]: https://www.eda2.com/conferenceHome/program/detail?key=s2
<!-- links -->