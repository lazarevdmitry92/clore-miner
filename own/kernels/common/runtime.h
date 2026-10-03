// Host runtime shared by the Pearl kernel libraries (TZ_v100_kernel_v2.md §5, TZ_sm80_kernel.md §3): NVML telemetry
// through dlopen (no header, no link stub), the online autotune state machine and the telemetry JSON for /summary.
#pragma once
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <string>
#include <vector>

namespace pearl {

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double env_f(const char *name, double def) {
    const char *v = getenv(name);
    return v ? atof(v) : def;
}

struct Nvml {
    bool ok = false;
    int (*byPci)(const char *, void **) = nullptr;
    int (*power)(void *, unsigned *) = nullptr;
    int (*clock)(void *, int, unsigned *) = nullptr;
    int (*throttle)(void *, unsigned long long *) = nullptr;
    Nvml() {
        void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!h) return;
        auto init = (int (*)())dlsym(h, "nvmlInit_v2");
        byPci = (int (*)(const char *, void **))dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2");
        power = (int (*)(void *, unsigned *))dlsym(h, "nvmlDeviceGetPowerUsage");
        clock = (int (*)(void *, int, unsigned *))dlsym(h, "nvmlDeviceGetClockInfo");
        throttle = (int (*)(void *, unsigned long long *))dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons");
        ok = init && byPci && power && clock && init() == 0;
    }
    void *device(const char *pci_bus_id) {   // as cudaDeviceGetPCIBusId prints it
        void *d = nullptr;
        return ok && byPci(pci_bus_id, &d) == 0 ? d : nullptr;
    }
};

inline Nvml &nvml() {
    static Nvml n;
    return n;
}

struct Telemetry { bool ok = false; unsigned mhz = 0, mw = 0; unsigned long long throttle = 0; };

inline Telemetry read_telemetry(void *dev) {
    Telemetry t;
    Nvml &n = nvml();
    if (!n.ok || !dev) return t;
    t.ok = n.clock(dev, 1 /* SM */, &t.mhz) == 0 && n.power(dev, &t.mw) == 0;
    if (n.throttle) n.throttle(dev, &t.throttle);
    return t;
}

// Online autotune: the best few variants of the startup timing are re-measured in turn while mining, each for
// PEARL_ONLINE_SECS (90) after PEARL_ONLINE_SETTLE (15) s, under the host's real power limit; the fastest is held for
// PEARL_ONLINE_HOLD (3600) s, then the round repeats. PEARL_ONLINE=0 turns it off; a pinned variant too.
struct OnlineTune {
    bool online = true, pinned = false;
    std::vector<int> cand;
    std::vector<double> rate;          // measured MAC/s per candidate in the last round
    int phase = -1;                    // candidate being measured; -1 = holding the winner
    double phase_t0 = 0, phase_macs = 0, phase_secs = 0, hold_until = 0;
    double recent_macs = 0, recent_secs = 0;   // decaying sums for k_actual

    void start(const std::vector<int> &candidates) {
        cand = candidates;
        rate.assign(cand.size(), 0.0);
        online = env_f("PEARL_ONLINE", 1) != 0 && !cand.empty();
        phase = online ? 0 : -1;
        phase_t0 = 0;
    }
    bool active() const { return online && !pinned; }
    const char *state() const { return !active() ? "off" : phase < 0 ? "hold" : "measuring"; }

    // variant for the next mining call
    int pick(int current) {
        if (!active()) return current;
        const double t = now_s();
        if (phase < 0) {
            if (t < hold_until) return current;
            phase = 0;
            phase_t0 = 0;
        }
        if (phase_t0 == 0) { phase_t0 = t; phase_macs = phase_secs = 0; }
        return cand[phase];
    }

    // account a call of variant v; returns the new winner at the end of a round, else -1
    int account(int v, double macs, double secs) {
        recent_macs = recent_macs * 0.9 + macs;
        recent_secs = recent_secs * 0.9 + secs;
        if (!active() || phase < 0 || cand[phase] != v) return -1;
        const double settle = env_f("PEARL_ONLINE_SETTLE", 15), dwell = env_f("PEARL_ONLINE_SECS", 90), t = now_s();
        if (t - phase_t0 >= settle) { phase_macs += macs; phase_secs += secs; }
        if (t - phase_t0 < settle + dwell) return -1;
        rate[phase] = phase_secs > 0 ? phase_macs / phase_secs : 0;
        phase_t0 = 0;
        if (++phase < (int)cand.size()) return -1;
        int best = 0;
        for (size_t i = 1; i < cand.size(); ++i)
            if (rate[i] > rate[best]) best = (int)i;
        phase = -1;
        hold_until = t + env_f("PEARL_ONLINE_HOLD", 3600);
        return cand[best];
    }
    double recent_rate() const { return recent_secs > 0 ? recent_macs / recent_secs : 0; }
};

// {"kernel","path","k_expected","k_actual","mac_s","sm_mhz","power_w","throttle":[...],"online_autotune"}
inline void telemetry_json(char *json, unsigned len, const std::string &kernel, const char *path, int sm,
                           double rate_expected, double mhz_expected, const OnlineTune &ot, void *nvdev) {
    const Telemetry t = read_telemetry(nvdev);
    const double k_exp = rate_expected > 0 && mhz_expected > 0 ? rate_expected / (sm * mhz_expected * 1e6) : 0;
    const double rate = ot.recent_rate(), k_act = t.ok && t.mhz ? rate / (sm * t.mhz * 1e6) : 0;
    std::string thr;
    const struct { unsigned long long bit; const char *name; } R[] = {
        {0x4, "power_cap"}, {0x8, "hw_slowdown"}, {0x20, "sw_thermal"}, {0x40, "hw_thermal"},
        {0x80, "hw_power_brake"}, {0x2, "app_clocks"}, {0x1, "idle"}};
    for (auto &x : R)
        if (t.throttle & x.bit) thr += std::string(thr.empty() ? "\"" : ",\"") + x.name + "\"";
    snprintf(json, len,
             "{\"kernel\":\"%s\",\"path\":\"%s\",\"k_expected\":%.1f,\"k_actual\":%.1f,\"mac_s\":%.4g,"
             "\"sm_mhz\":%u,\"power_w\":%.1f,\"throttle\":[%s],\"online_autotune\":\"%s\"}",
             kernel.c_str(), path, k_exp, k_act, rate, t.mhz, t.mw / 1000.0, thr.c_str(), ot.state());
}

}  // namespace pearl
