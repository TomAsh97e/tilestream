#pragma once

#include <string>

namespace tilestream {

[[nodiscard]] const char *version() noexcept;
[[nodiscard]] std::string info_json();

} // namespace tilestream
