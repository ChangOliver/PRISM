BUILD_DIR ?= build
CMAKE ?= cmake
JOBS ?= $(shell if command -v nproc >/dev/null 2>&1; then nproc; elif command -v sysctl >/dev/null 2>&1; then sysctl -n hw.ncpu; else echo 1; fi)

.PHONY: all clean

all:
	@mkdir -p $(BUILD_DIR)
	$(CMAKE) -S . -B $(BUILD_DIR)
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

clean:
	rm -rf $(BUILD_DIR)
