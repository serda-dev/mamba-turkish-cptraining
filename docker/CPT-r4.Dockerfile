# Preserve the pilot-tested CUDA, Transformers, Mamba and Arrow versions.
FROM serdadev/jamba2-cpt@sha256:5762cb5fffab046aca0c6c3a7a32fbc1733a8f51352b54244f738f0b6f4eb8e4
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libc6-dev rsync \
 && rm -rf /var/lib/apt/lists/*
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++
RUN printf '#include <stdlib.h>\nint main(void) {return EXIT_SUCCESS;}\n' > /tmp/compiler-check.c \
 && gcc /tmp/compiler-check.c -o /tmp/compiler-check \
 && /tmp/compiler-check \
 && rm /tmp/compiler-check.c /tmp/compiler-check
