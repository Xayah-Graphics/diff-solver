# syntax=docker/dockerfile:1
# hadolint global ignore=DL3007

FROM archlinux:latest AS build

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    pacman -Syu --noconfirm --needed base-devel cmake cuda ninja

WORKDIR /src
COPY --link CMakeLists.txt .
COPY --link solvers solvers
COPY --link examples examples

RUN . /etc/profile \
    && cmake -S . -B cmake-build-release -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${NVCC_CCBIN}" \
        -DCMAKE_CUDA_HOST_COMPILER="${NVCC_CCBIN}" \
        -DDIFF_SOLVER_BUILD_EXAMPLES=ON \
        -DDIFF_SOLVER_BUILD_SPECTRA_PROVIDERS=OFF \
    && cmake --build cmake-build-release --parallel 30


FROM archlinux:latest AS runtime

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    pacman -Syu --noconfirm --needed cuda gcc-libs \
    && groupadd --gid 10001 diff-solver \
    && useradd --uid 10001 --gid 10001 --home-dir /opt/diff-solver --shell /usr/bin/nologin diff-solver

COPY --from=build --chown=10001:10001 --link /src/cmake-build-release/examples/cloth/forward/diff-cloth-forward /opt/diff-solver/bin/
COPY --from=build --chown=10001:10001 --link /src/cmake-build-release/examples/smoke/forward/diff-smoke-forward /opt/diff-solver/bin/

USER 10001:10001
ENV HOME=/opt/diff-solver
ENV PATH="/opt/diff-solver/bin:${PATH}"
WORKDIR /opt/diff-solver

CMD ["diff-smoke-forward"]
