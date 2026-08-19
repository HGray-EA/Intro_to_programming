# ============================================================
# L&D CODE-ALONG: Data Science with Star Wars ⭐
# tidyverse edition — dplyr::starwars is built in, no download needed
# ============================================================
pacman::p_load(
  tidyverse, # includes data wrangling package and plotting package 
  broom, # tidymodels package
  ggrepel #labelling plots
  )

# ------------------------------------------------------------
# 1. LOAD THE DATA
# ------------------------------------------------------------
starwars # displays the dataset in your console

glimpse(starwars)
# Notice: films, vehicles, starships are LIST-COLUMNS (each cell holds a vector).
# height/mass are numeric, several categorical cols (species, sex, homeworld).

# Quick summary
summary(starwars$height)
summary(starwars$mass)

# ------------------------------------------------------------
# 2. BASIC WRANGLING WARM-UP
# ------------------------------------------------------------

# Humans over 1.8m tall
starwars %>%
  filter(species == "Human", height > 180) %>%
  select(name, height, mass, homeworld)

# Add BMI — a good excuse to talk about mutate()
starwars %>%
  mutate(bmi = mass / (height / 100)^2) %>%
  select(name, height, mass, bmi) %>%
  arrange(desc(bmi))
# 👀 Jabba the Hutt dominates — great segue into talking about outliers

# ------------------------------------------------------------
# 3. GROUP_BY + SUMMARIZE: the tidyverse bread and butter
# ------------------------------------------------------------

library(tidylog) # QA tool - Explains data transformations in english

starwars %>%
  filter(!is.na(species)) %>%
  group_by(species) %>%
  summarize(
    n = n(),
    mean_height = mean(height, na.rm = TRUE),
    mean_mass = mean(mass, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n > 1) %>%           # skip one-off species for a cleaner picture
  arrange(desc(mean_mass))

# ------------------------------------------------------------
# 4. CORRELATION: does height predict mass?
# ------------------------------------------------------------

sw_clean <- starwars %>%
  filter(!is.na(height), !is.na(mass))

cor(sw_clean$height, sw_clean$mass)
# Surprisingly weak/misleading — ask the group why before revealing:
# it's Jabba again, a single massive outlier dragging the correlation around.

cor.test(sw_clean$height, sw_clean$mass)

# Remove Jabba and recheck — nice "always plot your data" moment
sw_no_jabba <- sw_clean %>% filter(name != "Jabba Desilijic Tiure")
cor(sw_no_jabba$height, sw_no_jabba$mass)

# ------------------------------------------------------------
# 5. VISUALISE IT
# ------------------------------------------------------------

ggplot(sw_clean, aes(x = height, y = mass)) +
  geom_point(aes(color = species == "Human"), size = 3, alpha = 0.8) +
  geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
  geom_text_repel(
    data = sw_clean %>% filter(mass > 200),
    aes(label = name), size = 3
  ) +
  labs(
    title = "Height vs. Mass across the Star Wars universe",
    subtitle = "One outlier can wreck a correlation — always plot your data",
    x = "Height (cm)", y = "Mass (kg)",
    color = "Human?"
  ) +
  theme_minimal()

# Same plot without Jabba — regression line changes shape noticeably
ggplot(sw_no_jabba, aes(x = height, y = mass)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "steelblue") +
  labs(title = "Height vs. Mass (Jabba excluded)") +
  theme_minimal()

# ------------------------------------------------------------
# 6. SUPERVISED MACHINE LEARNING LINEAR MODEL + TIDY OUTPUT (broom)
# ------------------------------------------------------------

model <- lm(mass ~ height, data = sw_no_jabba)
summary(model)

# broom turns model objects into tidy tibbles — great for downstream ggplot/reporting
tidy(model)      # coefficients, std errors, p-values
glance(model)    # r.squared, AIC, etc. — one row per model

# Predict mass for a new hypothetical character
new_char <- tibble(height = 190)
predict(model, newdata = new_char)

# ------------------------------------------------------------
# 7. WORKING WITH LIST-COLUMNS: who's been in the most films?
# ------------------------------------------------------------

starwars %>%
  select(name, films) %>%
  mutate(n_films = map_int(films, length)) %>%
  arrange(desc(n_films)) %>%
  slice_head(n = 10) %>%
  ggplot(aes(x = fct_reorder(name, n_films), y = n_films)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(title = "Most-appearing Star Wars characters", x = NULL, y = "Films appeared in") +
  theme_minimal()

# unnest() to get one row per character-per-film (long format)
starwars %>%
  select(name, films) %>%
  unnest(films) %>%
  count(films, sort = TRUE) %>%
  rename(n_characters = n)

# ------------------------------------------------------------
# 8. UNSUPERVISED MACHINE LEARNING K-MEANS CLUSTERING
# Can unsupervised learning rediscover species-like groupings
# just from height, mass, and birth_year?
# ------------------------------------------------------------

cluster_data <- starwars %>%
  filter(!is.na(height), !is.na(mass), !is.na(birth_year)) %>%
  filter(name != "Jabba Desilijic Tiure")   # drop the outlier again

cluster_input <- cluster_data %>%
  select(height, mass, birth_year) %>%
  scale()   # IMPORTANT: standardize before clustering — different units/scales!

set.seed(42)
km <- kmeans(cluster_input, centers = 3, nstart = 25)

cluster_data <- cluster_data %>%
  mutate(cluster = factor(km$cluster))

ggplot(cluster_data, aes(x = height, y = mass, color = cluster)) +
  geom_point(size = 3) +
  geom_text_repel(aes(label = name), size = 2.8, max.overlaps = 15) +
  labs(
    title = "K-means clustering on height, mass, and birth year",
    subtitle = "3 clusters found with no knowledge of species — how well do they line up?",
    x = "Height (cm)", y = "Mass (kg)"
  ) +
  theme_minimal()

# Check: how do clusters map onto actual species?
cluster_data %>%
  count(cluster, species, sort = TRUE) %>%
  group_by(cluster) %>%
  slice_max(n, n = 3)

# BONUS talking point: try different values of `centers` (2, 4, 5) and
# discuss the elbow method / how you'd choose k without knowing species in advance.

# ------------------------------------------------------------
# 9. STRETCH GOAL FOR FAST FINISHERS
# ------------------------------------------------------------
# - Refit the lm() with species as a covariate: lm(mass ~ height + species)
# - Try PCA instead of raw variables: prcomp(cluster_input) then plot PC1 vs PC2
# - Build a t-test comparing mass between two homeworlds of your choice
# - Use eye_color or sex to explore chi-square / categorical association
