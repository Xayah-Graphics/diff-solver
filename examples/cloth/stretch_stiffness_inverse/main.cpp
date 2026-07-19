import std;
import xayah.cloth.data;
import xayah.examples.cloth.stretch_stiffness_inverse;
import xayah.solver;

namespace {

    constexpr float epsilon                       = 1.0e-2F;
    constexpr float finite_difference_tolerance   = 5.0e-3F;
    constexpr float adjoint_tolerance             = 1.0e-5F;
    constexpr float tape_tolerance                = 1.0e-6F;
    constexpr std::size_t optimization_iterations = 200u;

    xayah::cloth::Configuration make_grid(const std::size_t resolution, const float extent, const float time_step) {
        xayah::cloth::Configuration configuration{
            .rest_positions = std::vector<xayah::cloth::Vector3>(resolution * resolution),
            .triangles      = {},
            .anchors        = std::vector<std::optional<xayah::cloth::Vector3>>(resolution * resolution),
            .gravity        = {.x = 0.0F, .y = -9.81F, .z = 0.0F},
            .time_step      = time_step,
        };
        for (std::size_t row = 0; row < resolution; ++row) {
            for (std::size_t column = 0; column < resolution; ++column) {
                const std::size_t particle             = row * resolution + column;
                configuration.rest_positions[particle] = {
                    .x = -extent * 0.5F + extent * static_cast<float>(column) / static_cast<float>(resolution - 1),
                    .y = 0.0F,
                    .z = extent * static_cast<float>(row) / static_cast<float>(resolution - 1),
                };
                if (row == 0) configuration.anchors[particle] = configuration.rest_positions[particle];
            }
        }
        configuration.triangles.reserve((resolution - 1) * (resolution - 1) * 2);
        for (std::size_t row = 0; row + 1 < resolution; ++row) {
            for (std::size_t column = 0; column + 1 < resolution; ++column) {
                const std::uint32_t top_left     = static_cast<std::uint32_t>(row * resolution + column);
                const std::uint32_t top_right    = top_left + 1;
                const std::uint32_t bottom_left  = top_left + static_cast<std::uint32_t>(resolution);
                const std::uint32_t bottom_right = bottom_left + 1;
                if ((row + column) % 2 == 0) {
                    configuration.triangles.push_back({.first = top_left, .second = bottom_left, .third = bottom_right});
                    configuration.triangles.push_back({.first = top_left, .second = bottom_right, .third = top_right});
                } else {
                    configuration.triangles.push_back({.first = top_left, .second = bottom_left, .third = top_right});
                    configuration.triangles.push_back({.first = top_right, .second = bottom_left, .third = bottom_right});
                }
            }
        }
        return configuration;
    }

} // namespace

int main() {
    xayah::cloth::examples::stretch_stiffness_inverse::StretchStiffnessInverseTask task(make_grid(16u, 2.0F, 1.0F / 240.0F));
    const xayah::cloth::examples::stretch_stiffness_inverse::StretchStiffnessGradientCheck store_all = task.check_gradient(xayah::solver::TapeMode::store_all, epsilon);
    const xayah::cloth::examples::stretch_stiffness_inverse::StretchStiffnessGradientCheck recompute = task.check_gradient(xayah::solver::TapeMode::recompute_step_cache, epsilon);

    std::println("CUDA cloth stretch-stiffness inverse gradient checks");
    for (const auto [name, metrics] : std::array{
             std::pair<std::string_view, xayah::cloth::examples::stretch_stiffness_inverse::StretchStiffnessGradientCheck>{"store_all", store_all},
             std::pair<std::string_view, xayah::cloth::examples::stretch_stiffness_inverse::StretchStiffnessGradientCheck>{"recompute_step_cache", recompute},
         }) {
        const double finite_difference_error = std::abs(metrics.finite_difference - metrics.jvp_inner_product) / std::max({1.0e-12, std::abs(metrics.finite_difference), std::abs(metrics.jvp_inner_product)});
        const double adjoint_error           = std::abs(metrics.jvp_inner_product - metrics.vjp_inner_product) / std::max({1.0e-12, std::abs(metrics.jvp_inner_product), std::abs(metrics.vjp_inner_product)});
        std::println("  {}: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, fd/jvp={:.3e}, jvp/vjp={:.3e}", name, metrics.finite_difference, metrics.jvp_inner_product, metrics.vjp_inner_product, finite_difference_error, adjoint_error);
        if (finite_difference_error > finite_difference_tolerance) throw std::runtime_error(std::format("{} inverse finite-difference check failed", name));
        if (adjoint_error > adjoint_tolerance) throw std::runtime_error(std::format("{} inverse adjoint check failed", name));
    }
    const double tape_difference = std::max({std::abs(store_all.finite_difference - recompute.finite_difference), std::abs(store_all.jvp_inner_product - recompute.jvp_inner_product), std::abs(store_all.vjp_inner_product - recompute.vjp_inner_product)});
    std::println("  tape consistency: max_abs_difference={:.9e}", tape_difference);
    if (tape_difference > tape_tolerance) throw std::runtime_error("inverse tape consistency check failed");

    std::println("\nCUDA cloth stretch-stiffness inversion");
    std::println("  iteration {:3}: stiffness={:.6f}, loss={:.9e}, loss_ratio={:.3e}, gradient={:.9e}", task.metrics.iteration, task.metrics.stretch_stiffness, task.metrics.loss, 1.0, task.metrics.log_stiffness_gradient);
    for (std::size_t iteration = 0u; iteration < optimization_iterations; ++iteration) {
        task.optimize_step();
        if (task.metrics.iteration % 10u == 0u) std::println("  iteration {:3}: stiffness={:.6f}, loss={:.9e}, loss_ratio={:.3e}, gradient={:.9e}", task.metrics.iteration, task.metrics.stretch_stiffness, task.metrics.loss, task.metrics.loss / task.metrics.initial_loss, task.metrics.log_stiffness_gradient);
    }
    const double stiffness_error = std::abs(static_cast<double>(task.metrics.stretch_stiffness) - task.options.target_stretch_stiffness) / task.options.target_stretch_stiffness;
    const double loss_ratio      = task.metrics.loss / task.metrics.initial_loss;
    std::println("  relative_stiffness_error={:.3e}, final/initial_loss={:.3e}", stiffness_error, loss_ratio);
    if (stiffness_error > 1.0e-2) throw std::runtime_error("stretch-stiffness recovery check failed");
    if (loss_ratio > 1.0e-4) throw std::runtime_error("stretch-stiffness loss reduction check failed");
    return 0;
}
