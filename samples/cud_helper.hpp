#pragma once

std::pair<float, float> meanStd(const std::vector<float>& data) {
    float mean = 0.0;
    float acc2 = 0.0;

    for (size_t i = 0; i < data.size(); ++i) {
        float delta = data[i] - mean;
        mean += delta / float(i + 1);
        float delta2 = data[i] - mean;
        acc2 += delta * delta2;
    }

    float variance = acc2 / (float)data.size();
    return {mean, std::sqrt(variance)};
}