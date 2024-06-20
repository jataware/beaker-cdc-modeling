# Description
Initialize the model components, including initial infections, latent infections, hospitalization processes, and observation processes.

# Code
```

# Initializing model components:

# 1) A deterministic generation time
pmf_array = jnp.array([0.25, 0.25, 0.25, 0.25])
gen_int = DeterministicPMF(pmf_array, name="gen_int")

# 2) Initial infections
I0 = InfectionSeedingProcess(
    "I0_seeding",
    DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
    SeedInfectionsZeroPad(pmf_array.size),
)

# 3) The latent infections process
latent_infections = Infections()

# 4) The latent hospitalization process:

# First, define a deterministic infection to hosp pmf
inf_hosp_int = DeterministicPMF(
    jnp.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.1, 0.1, 0.05]
    ),
    name="inf_hosp_int",
)

latent_admissions = HospitalAdmissions(
    infection_to_admission_interval_rv=inf_hosp_int,
    infect_hosp_rate_rv=DistributionalRV(
        dist=dist.LogNormal(jnp.log(0.05), 0.05), name="IHR"
    ),
)

# 5) An observation process for the hospital admissions
admissions_process = PoissonObservation()

# 6) A random walk process (it could be deterministic using
# pyrenew.process.DeterministicProcess())
Rt_process = RtRandomWalkProcess(
    Rt0_dist = dist.TruncatedNormal(loc=1.2, scale=0.2, low=0),
    Rt_transform = t.ExpTransform().inv,
    Rt_rw_dist = dist.Normal(0, 0.025),

```
