using 
    CSV, 
    DataFrames, 
    MixedModels, 
    PrettyTables, 
    Statistics,
    Turing, 
    CategoricalArrays, 
    StatsModels, 
    LinearAlgebra

# loading and transforming data
file = "_data/ifsc_boulder_results_2025.csv"
data = CSV.read(file, DataFrame)
select!(data, Not(1)) # first col is just an index

data.comp_id = categorical(data.comp_id)
data.athlete_id = categorical(data.athlete_id)
data.athlete_country = categorical(data.athlete_country)

unstacked_scores = unstack(data[:, ["comp_id", "athlete_id", "round", "score"]], :round, :score)
select!(data, Not([:round, :score]))
unique!(data)
data = innerjoin(data, unstacked_scores, on = [:comp_id, :athlete_id])

rename!(data, "Qualification" => "score_quali", "Semi-final" => "score_semi", "Final" => "score_final")

data.comp_idx = levelcode.(data.comp_id)
data.athlete_idx = levelcode.(data.athlete_id)

# let's take out one event for testing out model later. This is a tiny test dataset, but that's ok. We are just having fun!
test_comp = 1408
test_data = data[data.event_id .== test_comp, :]
data = data[data.event_id .!= test_comp, :]


# save processed data to csv
# output_file = joinpath("_data", "processed_ifsc_boulder_results_2025.csv")
# output_data = select(data, Not([:dcat_id, :first_season, :comp_idx, :athlete_idx]))
# CSV.write(output_file, output_data)

@model function bayesian_multilevel_model(data)
    N = length(data.score_final)
    Ncomp = length(levels(data.comp_id))
    Nath = length(levels(data.athlete_id))

    # Hyperpriors
    σ ~ truncated(Normal(0, 50), 0, Inf)
    σ_comp ~ truncated(Normal(0, 50), 0, Inf)
    σ_ath ~ truncated(Normal(0, 50), 0, Inf)
    
    # Fixed effects
    α ~ Normal(0, 100)
    β_semi ~ Normal(0, 10)
    β_quali ~ Normal(0, 10)
    
    # Varying intercepts for competitions and athletes
    comp_eff ~ filldist(Normal(0, σ_comp), Ncomp)
    ath_eff ~ filldist(Normal(0, σ_ath), Nath)

    # Likelihood
    for i in 1:N
        μ = α +
            β_semi * data.score_semi[i] +
            β_quali * data.score_quali[i] +
            comp_eff[data.comp_idx[i]] +
            ath_eff[data.athlete_idx[i]]
        
        data.score_final[i] ~ Normal(μ, σ)
    end
end

# Instantiate and sample
model = bayesian_multilevel_model(data)
chain = sample(model, NUTS(), 4_000)

describe(chain)

function predict_score_final(chain, test_data, train_data)

    α = mean(chain, :α)
    β_semi = mean(chain, :β_semi)
    β_quali = mean(chain, :β_quali)

    comp_eff_names = MCMCChains.namesingroup(chain, :comp_eff)
    ath_eff_names = MCMCChains.namesingroup(chain, :ath_eff)

    # comp_eff = [mean(chain, k) for k in comp_eff_names]
    # ath_eff = [map_estimates[k] for k in ath_eff_names]
    # println(keys(comp_eff))

    comp_levels = levels(train_data.comp_id)
    ath_levels = levels(train_data.athlete_id)

    comp_codes = levelcode.(CategoricalArray(test_data.comp_id, levels=comp_levels))
    ath_codes = levelcode.(CategoricalArray(test_data.athlete_id, levels=ath_levels))

    semi_scores = test_data[!, :score_semi]
    quali_scores = test_data[!, :score_quali]

    n = nrow(test_data)
    preds = Vector{Float64}(undef, n)

    for i in 1:n
        semi = semi_scores[i]
        quali = quali_scores[i]
        comp_id = comp_codes[i]
        ath_id = ath_codes[i]

        comp_eff = mean(chain, Symbol("comp_eff[$comp_id]"))
        ath_eff = mean(chain, Symbol("ath_eff[$ath_id]"))

        comp_term = comp_id ≤ length(comp_eff_names) ? comp_eff : 0.0
        ath_term = ath_id ≤ length(ath_eff_names) ? ath_eff : 0.0

        preds[i] = α + β_semi * semi + β_quali * quali + comp_term + ath_term
    end

    return preds
end

# map_estimate = maximum_a_posteriori(model)
preds = predict_score_final(chain, data, data)

data.preds = preds
errors = data.score_final .- data.preds

mae = mean(abs.(errors))
rmse = sqrt(mean(errors .^ 2))

test_data.pred = predict_score_final(chain, test_data, data)
test_data.pred = round.(test_data.pred, digits=1)
test_data.residual = round.(test_data.score_final .- test_data.pred,digits=1)

test_output = test_data[:, [:event_name, :gender, :athlete_name, :score_final, :pred, :residual]]

show(test_output, allrows=true)

mean(chain, :σ_ath)^2

# let's plot the distributions for the intercepts of Sorato and Janja. 
# Janja did not participate at many comps this season, so we would expect wider tails
sorato_samples = Array(chain[:"ath_eff[34]"])
janja_samples = Array(chain[:"ath_eff[9]"])

# Plot densities
density(sorato_samples, label = "Sorato", linewidth = 2)
vline!([mean(sorato_samples)], label = "Mean Sorato", linestyle = :dash, color = :blue)

density!(janja_samples, label = "Janja", linewidth = 2)
vline!([mean(janja_samples)], label = "Mean Janja", linestyle = :dash, color = :red)