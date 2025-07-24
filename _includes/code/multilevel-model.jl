using 
    CSV, 
    DataFrames, MixedModels, PrettyTables, Statistics,
    CategoricalArrays

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

# modelling with MixedModels.jl (frequentist methods)
formula = @formula(
    score_final ~ 1 + gender + athlete_country + first_season
    + (1 + score_quali + score_semi | athlete_country/athlete_id) 
    + (1 + score_quali + score_semi | comp_id)
)


model = fit(LinearMixedModel, formula, data)

# Display the model summary
println(model)

# Calculate the Mean Squared Error (MSE)
predicted = predict(model)
actual = data.score_final
residual = predicted .- actual
mse = mean((predicted .- actual).^2)

results = select(data, ["event_id", "comp_id", "athlete_name", "score_final"])
results.predicted = predicted
results.residual = residual

for event_id in unique(results.event_id)
    println(pretty_table(results[(results.event_id .== 1405) , :]))
end

println("Mean Squared Error (MSE): ", mse)


newdata = DataFrame(
    event_id = [9999],
    event_name = ["NEW"],
    dcat_id = [7],
    dcat = ["BOULDER Women"],
    athlete_name = ["NEW"],
    birthday = nothing,
    score_final = [0.0],                 # dummy value, will be ignored in prediction
    gender = ["female"],
    athlete_country = ["JPN"],
    first_season = [2019],
    score_quali = [110.0],
    score_semi = [75.0],
    athlete_id = [9999],                 # can be new
    comp_id = [99999]            # new competition
)
# Ensure grouping variables are categorical with the same levels
data.comp_id = categorical(data.comp_id, levels=levels(data.comp_id))
data.athlete_id = categorical(data.athlete_id, levels=levels(data.athlete_id))
data.athlete_country = categorical(data.athlete_country, levels=levels(data.athlete_country))

combined_data = vcat(data, newdata)
# levels!(data.comp_id, vcat(levels(data.comp_id), 99999))
levels!(data.athlete_id, vcat(levels(data.athlete_id), 9999))

predict(model, newdata; new_re_levels=:missing)
println("Predicted score_final: ", ŷ)
