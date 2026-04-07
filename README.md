<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/MetaX-MACA/vLLM-metax/master/docs/assets/logos/vllm-metax-logo.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM MetaX Plugin
</h3>

<p align="center">
| <a href="https://www.metax-tech.com/en/"><b>About MetaX</b></a> | <a href="https://vllm-metax.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://slack.vllm.ai"><b>#sig-maca</b></a> </a> |
</p>

---

*Latest News* 🔥
- [2026/3] Released vllm-metax **v0.15.0** 🦐 — aligned with vLLM *v0.15.0*, more models and more features!
- [2026/3] Released vllm-metax **v0.14.0** 🚀 — aligned with vLLM *v0.14.0*, same as usual.
- [2026/2] Released vllm-metax **v0.13.0** 🧨 — aligned with vLLM *v0.13.0*, brings you the latest features and model in v0.13.0!
- [2026/1] Released vllm-metax **v0.12.0** 😎 — aligned with vLLM *v0.12.0*, supported more models and improved performance.
- [2026/1] Released vllm-metax **v0.11.2** 👻 — aligned with vLLM *v0.11.2*, supported more models and improved performance.
- [2025/11] Released vllm-metax **v0.10.2** 🎉 — aligned with vLLM *v0.10.2*, improved model performance, and fixed key decoding bugs.
- [2025/11] We hosted [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/xSrYXjNgr1HbCP4ExYNG1w) focusing on distributed inference and diverse accelerator support with vLLM! Please find the meetup slides [here](https://drive.google.com/drive/folders/1nQJ8ZkLSjKxvu36sSHaceVXtttbLvvu-?usp=drive_link).
- [2025/08] We hosted [vLLM Shanghai Meetup](https://mp.weixin.qq.com/s/pDmAXHcN7Iqc8sUKgJgGtg) focusing on building, developing, and integrating with vLLM! Please find the meetup slides [here](https://drive.google.com/drive/folders/1OvLx39wnCGy_WKq8SiVKf7YcxxYI3WCH).


## About

vLLM MetaX is a hardware plugin that enables vLLM to run seamlessly on MetaX GPUs. MetaX provides a cuda-like backend through [*MACA*](https://www.metax-tech.com/en/goods/platform.html?cid=4), delivering a near-native CUDA experience on MetaX hardware.

It is the recommended approach for supporting the MetaX backend within the vLLM community. 

The plugin is implemented in accordance with the vLLM plugin RFCs:
 - [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)
 - [[RFC]: Enhancing vLLM Plugin Architecture](https://github.com/vllm-project/vllm/issues/19161)

These RFCs help ensure proper feature and functionality support when integrating MetaX GPUs with vLLM.

## Prerequisites

- Hardware: MetaX C-series
- OS: Linux
- Software:
  - Python >= 3.10, <= 3.12
  - vLLM (the same version as vllm-metax)
  - Docker support

## Getting Started

vLLM MetaX currently supports deployment only with Docker images released by the [MetaX developer community](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm-metax), which work out of the box. 

> The Dockerfile for other OS environments is still under testing.

If you want to develop, debug, or test the latest features in vllm-metax, you may need to build it from source. Please follow this [*source build tutorial*](https://vllm-metax.readthedocs.io/en/v0.13.0/getting_started/installation/maca.html).

## Branch

vllm-metax has three kinds of branches.

- **master**: the main branch, which tracks the upstream vLLM main branch.
- **vX.Y.Z-dev**: development branches created after a vLLM release.
  > For example, `v0.1x.0-dev` is the development branch for a newly released branch like `releases/v0.1x.0`.
- **releases/vX.Y.Z**: release branches created from `v0.1x.0-dev`, indicating that the corresponding vllm-metax development branch has been fully tested and released.
  > For example, vllm-metax's `releases/v0.1x.0` corresponds to vLLM's `releases/v0.1x.0`. The same naming rule applies to tags.

Below are the maintained branches:

| Branch      | Status       | Note                                 |
|-------------|--------------|--------------------------------------|
| master      | N/A | Tracks vLLM main; functionality is not guaranteed |
| v0.19.0-dev | N/A | WIP |
| v0.18.0-dev | N/A | WIP |
| v0.17.0-dev | N/A | Planned for release in April |
| v0.16.0 | N/A | **Skipped** |
| releases/v0.15.0 | Released | Corresponds to vLLM release v0.15.0 |
| releases/v0.14.0 | Released | Corresponds to vLLM release v0.14.0 |
| releases/v0.13.0 | Released | Corresponds to vLLM release v0.13.0 |
| releases/v0.12.0 | Released | Corresponds to vLLM release v0.12.0 |
| releases/v0.11.2 | Released | Corresponds to vLLM release v0.11.2 |
| releases/v0.10.2 | Released | Corresponds to vLLM release v0.10.2 |


For more details, please check the [Quickstart Guide](https://vllm-metax.readthedocs.io/en/latest/getting_started/quickstart.html).

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.