# Date: 4/15/21
# Authors:Alec Soudry

# Ebay Auctions Discriminant Analysis

eBayAuctions.df <- read.csv("eBayAuctions.csv", stringsAsFactors = TRUE)

# convert to factor variable #
eBayAuctions.df$Duration <- as.factor(eBayAuctions.df$Duration)

# create dummy variables #
str(eBayAuctions.df)
library(fastDummies)
eBayAuctions.df <- dummy_cols(eBayAuctions.df, 
                              select_columns = c("Category","currency","Duration","endDay"),
                              remove_first_dummy = TRUE, remove_selected_columns = TRUE)
str(eBayAuctions.df)
t(t(names(eBayAuctions.df)))

# remove ClosePrice #
eBayAuctions.df <- eBayAuctions.df[, -2]
t(t(names(eBayAuctions.df)))

# partition data #
set.seed(1)
train.index <- sample(rownames(eBayAuctions.df), nrow(eBayAuctions.df) * 0.6)
train.df <- eBayAuctions.df[train.index, ]
valid.index <- setdiff(rownames(eBayAuctions.df), train.index)
valid.df <- eBayAuctions.df[valid.index, ]

####### full logistic regression model #######
# use glm() (generalized linear model) with family = "binomial" to fit a logistic
# regression
logit.reg <- glm(Competitive. ~ ., data = train.df, family = "binomial")
options(scipen = 999)
summary(logit.reg)

# use predict() with type = "response" to compute predicted probabilities
logit.reg.pred <- predict(logit.reg,                
                          valid.df,                 
                          type = "response")        

data.frame(actual = valid.df$Competitive.[1:20], predicted = logit.reg.pred[1:20])

####### roc curve #######
library(pROC)
r <- roc(valid.df$Competitive., logit.reg.pred)
plot.roc(r)
auc(r)

####### lift chart #######
library(gains)
gain <- gains(valid.df$Competitive., logit.reg.pred, groups = length(logit.reg.pred))
gain
# plot lift chart
plot(c(0, gain$cume.pct.of.total * sum(valid.df$Competitive.)) ~ c(0, gain$cume.obs),
     xlab = "# of auctions", ylab = "Cumulative", main = "Lift Chart", type = "l")
lines(c(0, sum(valid.df$Competitive.)) ~ c(0, nrow(valid.df)), lty = 2)

####### decile lift chart #######
gain <- gains(valid.df$Competitive., logit.reg.pred)
heights <- gain$mean.resp / mean(valid.df$Competitive)
dec.lift <- barplot(heights, names.arg = gain$depth, ylim = c(0, 9),
                    xlab = "Percentile", ylab = "Mean Competitiveness", main = "Decile-wise Lift Chart")
gain

# discriminant analysis #
library(DiscriMiner)
eBayAuctions.da <- linDA(eBayAuctions.df[, -3],
                         eBayAuctions.df$Competitive.,
                         validation = "learntest",
                         learn = as.numeric(train.index),
                         test = as.numeric(valid.index))
options(scipen = 999, digits = 5)
eBayAuctions.da$functions

library(caret)
confusionMatrix(eBayAuctions.da$classification, 
                as.factor(valid.df$Competitive.), 
                positive = "1")


#####
# Ebay Auctions k nearest neighbors

# partition the data
set.seed(1)
train.index <- sample(rownames(eBayAuctions.df), nrow(eBayAuctions.df) * 0.6)
valid.index <- as.numeric(setdiff(rownames(eBayAuctions.df), train.index))
eBayAuctions.train <- eBayAuctions.df[train.index, ]
eBayAuctions.valid <- eBayAuctions.df[valid.index, ]

train.norm <- eBayAuctions.train
valid.norm <- eBayAuctions.train

# normalize numerical predictors to 0-1 scale
cols <- colnames(train.norm[ , c(2)])
for (i in cols) {
  valid.norm[[i]] <- (housing.valid[[i]] - min(housing.train[[i]])) / (max(housing.train[[i]]) - min(housing.train[[i]]))
  train.norm[[i]] <- (housing.train[[i]] - min(housing.train[[i]])) / (max(housing.train[[i]]) - min(housing.train[[i]]))
}

summary(train.norm)
summary(valid.norm)

# classifying using k=1
library(FNN)
eBayAuctions.nn <- knn(train = train.norm[, -c(3)], 
                       test = valid.norm[, -c(3)],
                       cl = train.norm$Competitive.,
                       k = 1)

# look at the confusion matrix
library(caret)
confusionMatrix(eBayAuctions.nn, as.factor(valid.norm$Competitive.), positive = "1")

# initialize a data frame with two columns: k and accuracy
accuracy.df <- data.frame(k = seq(1, 30, 1), accuracy = rep(0, 30))

# compute knn for different k on validation set
for (i in 1:30) {
  eBayAuctions.knn.pred <- knn(train = train.norm[, -c(3)], 
                               test = valid.norm[, -c(3)],
                               cl = train.norm$Competitive., 
                               k = i)
  accuracy.df[i, 2] <- confusionMatrix(eBayAuctions.knn.pred, as.factor(valid.norm$Competitive.), positive = "1")$overall[1]
}
accuracy.df

# optimal k
knn.pred.best <- knn(train.norm[, -c(3)], 
                     valid.norm[, -c(3)], 
                     cl = train.norm$Competitive., 
                     k = 1, 
                     prob = TRUE)                     
confusionMatrix(knn.pred.best, as.factor(valid.norm$Competitive.), positive = "1")

#####
# IMDB k nearest neighbors
# read in IMDB data
IMDB.df <- read.csv("IMDB Movie Dataset.csv", 
                    fileEncoding="UTF-8-BOM",
                    stringsAsFactors = TRUE)

# find and remove duplicate records
IMDB.df[duplicated(IMDB.df), ]
IMDB.df <- unique(IMDB.df)

dim(IMDB.df)
str(IMDB.df)

# plot keywords
head(IMDB.df$plot_keywords)
class(IMDB.df$plot_keywords)

# parse out the keywords from the pipe-delimited string and determine keyword frequency
parse_key <- data.frame(table(unlist(strsplit(as.character(IMDB.df$plot_keywords), split = "|",
                                              fixed = TRUE))))

# list the 20 most frequent keywords
head(parse_key[order(parse_key$Freq, decreasing = TRUE), ], 20)

# create binary dummy variables for top 20 keywords
IMDB.df$key_love <- ifelse(grepl("love", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_friend <- ifelse(grepl("friend", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_murder <- ifelse(grepl("murder", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_death <- ifelse(grepl("death", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_police <- ifelse(grepl("police", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_new_york_city <- ifelse(grepl("new york city", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_high_school <- ifelse(grepl("high school", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_alien <- ifelse(grepl("alien", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_school <- ifelse(grepl("school", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_boy <- ifelse(grepl("boy", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_fbi <- ifelse(grepl("fbi", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_revenge <- ifelse(grepl("revenge", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_friendship <- ifelse(grepl("friendship", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_drugs <- ifelse(grepl("drugs", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_prison <- ifelse(grepl("prison", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_money <- ifelse(grepl("money", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_marriage <- ifelse(grepl("marriage", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_dog <- ifelse(grepl("dog", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_female <- ifelse(grepl("female protagonist", IMDB.df$plot_keywords), 1, 0)
IMDB.df$key_island <- ifelse(grepl("island", IMDB.df$plot_keywords), 1, 0)

# genre 
head(IMDB.df$genre)

# parse out the genres from the pipe-delimited string and determine genre frequency
parse_genre <- data.frame(table(unlist(strsplit(as.character(IMDB.df$genres), split = "|",
                                                fixed = TRUE))))
# list all parsed genres
parse_genre[order(parse_genre$Freq, decreasing = TRUE), ]

# create binary dummy variables for parsed genres
IMDB.df$genre_drama <- ifelse(grepl("Drama", IMDB.df$genres), 1, 0)
IMDB.df$genre_comedy <- ifelse(grepl("Comedy", IMDB.df$genres), 1, 0)
IMDB.df$genre_thriller <- ifelse(grepl("Thriller", IMDB.df$genres), 1, 0)
IMDB.df$genre_action <- ifelse(grepl("Action", IMDB.df$genres), 1, 0)
IMDB.df$genre_romance <- ifelse(grepl("Romance", IMDB.df$genres), 1, 0)
IMDB.df$genre_adventure <- ifelse(grepl("Adventure", IMDB.df$genres), 1, 0)
IMDB.df$genre_crime <- ifelse(grepl("Crime", IMDB.df$genres), 1, 0)
IMDB.df$genre_scifi <- ifelse(grepl("Sci-Fi", IMDB.df$genres), 1, 0)
IMDB.df$genre_fantasy <- ifelse(grepl("Fantasy", IMDB.df$genres), 1, 0)
IMDB.df$genre_horror <- ifelse(grepl("Horror", IMDB.df$genres), 1, 0)
IMDB.df$genre_family <- ifelse(grepl("Family", IMDB.df$genres), 1, 0)
IMDB.df$genre_mystery <- ifelse(grepl("Mystery", IMDB.df$genres), 1, 0)
IMDB.df$genre_biography <- ifelse(grepl("Biography", IMDB.df$genres), 1, 0)
IMDB.df$genre_animation <- ifelse(grepl("Animation", IMDB.df$genres), 1, 0)
IMDB.df$genre_music <- ifelse(grepl("Music", IMDB.df$genres), 1, 0)
IMDB.df$genre_war <- ifelse(grepl("War", IMDB.df$genres), 1, 0)
IMDB.df$genre_history <- ifelse(grepl("History", IMDB.df$genres), 1, 0)
IMDB.df$genre_sport <- ifelse(grepl("Sport", IMDB.df$genres), 1, 0)
IMDB.df$genre_musical <- ifelse(grepl("Musical", IMDB.df$genres), 1, 0)
IMDB.df$genre_documentary <- ifelse(grepl("Documentary", IMDB.df$genres), 1, 0)
IMDB.df$genre_western <- ifelse(grepl("Western", IMDB.df$genres), 1, 0)

# director
# Find the most frequent directors
directorcount <- aggregate(IMDB.df$movie_title, by = list(IMDB.df$director_name), FUN = length)
names(directorcount) <- c("DirectorName", "MovieCount")
head(directorcount[order(directorcount$MovieCount, decreasing = TRUE), ], 20)

# Create dummy variables for directors with at least 15 films
IMDB.df$director_spielberg <- ifelse(IMDB.df$director_name == "Steven Spielberg", 1, 0)
IMDB.df$director_allen <- ifelse(IMDB.df$director_name == "Woody Allen", 1, 0)
IMDB.df$director_eastwood <- ifelse(IMDB.df$director_name == "Clint Eastwood", 1, 0)
IMDB.df$director_scorsese <- ifelse(IMDB.df$director_name == "Martin Scorsese", 1, 0)
IMDB.df$director_scott <- ifelse(IMDB.df$director_name == "Ridley Scott", 1, 0)
IMDB.df$director_lee <- ifelse(IMDB.df$director_name == "Spike Lee", 1, 0)
IMDB.df$director_soderbergh <- ifelse(IMDB.df$director_name == "Steven Soderbergh", 1, 0)
IMDB.df$director_burton <- ifelse(IMDB.df$director_name == "Tim Burton", 1, 0)
IMDB.df$director_harlin <- ifelse(IMDB.df$director_name == "Renny Harlin", 1, 0)

# actor 1 
# Find the most frequent actor 1s
actor1count <- aggregate(IMDB.df$movie_title, by = list(IMDB.df$actor_1_name), FUN = length)
names(actor1count) <- c("Actor1Name", "MovieCount")
head(actor1count[order(actor1count$MovieCount, decreasing = TRUE), ], 30)

# Create dummy variables for actors with at least 20 films
IMDB.df$actor_1_deniro <- ifelse(IMDB.df$actor_1_name == "Robert De Niro", 1, 0)
IMDB.df$actor_1_depp <- ifelse(IMDB.df$actor_1_name == "Johnny Depp", 1, 0)
IMDB.df$actor_1_cage <- ifelse(IMDB.df$actor_1_name == "Nicolas Cage", 1, 0)
IMDB.df$actor_1_simmons <- ifelse(IMDB.df$actor_1_name == "J.K. Simmons", 1, 0)
IMDB.df$actor_1_willis <- ifelse(IMDB.df$actor_1_name == "Bruce Willis", 1, 0)
IMDB.df$actor_1_washington <- ifelse(IMDB.df$actor_1_name == "Denzel Washington", 1, 0)
IMDB.df$actor_1_damon <- ifelse(IMDB.df$actor_1_name == "Matt Damon", 1, 0)
IMDB.df$actor_1_neeson <- ifelse(IMDB.df$actor_1_name == "Liam Neeson", 1, 0)
IMDB.df$actor_1_ford <- ifelse(IMDB.df$actor_1_name == "Harrison Ford", 1, 0)
IMDB.df$actor_1_williams <- ifelse(IMDB.df$actor_1_name == "Robin Williams", 1, 0)
IMDB.df$actor_1_buscemi <- ifelse(IMDB.df$actor_1_name == "Steve Buscemi", 1, 0)
IMDB.df$actor_1_murray <- ifelse(IMDB.df$actor_1_name == "Bill Murray", 1, 0)
IMDB.df$actor_1_statham <- ifelse(IMDB.df$actor_1_name == "Jason Statham", 1, 0)
IMDB.df$actor_1_downey <- ifelse(IMDB.df$actor_1_name == "Robert Downey Jr.", 1, 0)
IMDB.df$actor_1_freeman <- ifelse(IMDB.df$actor_1_name == "Morgan Freeman", 1, 0)
IMDB.df$actor_1_cruise <- ifelse(IMDB.df$actor_1_name == "Tom Cruise", 1, 0)
IMDB.df$actor_1_reeves <- ifelse(IMDB.df$actor_1_name == "Keanu Reeves", 1, 0)
IMDB.df$actor_1_hanks <- ifelse(IMDB.df$actor_1_name == "Tom Hanks", 1, 0)
IMDB.df$actor_1_bale <- ifelse(IMDB.df$actor_1_name == "Christian Bale", 1, 0)
IMDB.df$actor_1_butler <- ifelse(IMDB.df$actor_1_name == "Gerard Butler", 1, 0)
IMDB.df$actor_1_spacey <- ifelse(IMDB.df$actor_1_name == "Kevin Spacey", 1, 0)
IMDB.df$actor_1_johansson <- ifelse(IMDB.df$actor_1_name == "Scarlett Johansson", 1, 0)
IMDB.df$actor_1_hopkins <- ifelse(IMDB.df$actor_1_name == "Anthony Hopkins", 1, 0)
IMDB.df$actor_1_jackman<- ifelse(IMDB.df$actor_1_name == "Hugh Jackman", 1, 0)
IMDB.df$actor_1_dicaprio <- ifelse(IMDB.df$actor_1_name == "Leonardo DiCaprio", 1, 0)
IMDB.df$actor_1_stallone <- ifelse(IMDB.df$actor_1_name == "Sylvester Stallone", 1, 0)
IMDB.df$actor_1_mcconaughey <- ifelse(IMDB.df$actor_1_name == "Matthew McConaughey", 1, 0)
IMDB.df$actor_1_hoffman <- ifelse(IMDB.df$actor_1_name == "Philip Seymour Hoffman", 1, 0)
IMDB.df$actor_1_ferrell <- ifelse(IMDB.df$actor_1_name == "Will Ferrell", 1, 0)

# actor 2
# Find the most frequent actor 2s
actor2count <- aggregate(IMDB.df$movie_title, by = list(IMDB.df$actor_2_name), FUN = length)
names(actor2count) <- c("Actor2Name", "MovieCount")
head(actor2count[order(actor2count$MovieCount, decreasing = TRUE), ], 30)

# Create dummy variables for actors with at least 15 films
IMDB.df$actor_2_freeman <- ifelse(IMDB.df$actor_2_name == "Morgan Freeman", 1, 0)
IMDB.df$actor_2_theron <- ifelse(IMDB.df$actor_2_name == "Charlize Theron", 1, 0)

# actor 3
# Find the most frequent actor 3s
actor3count <- aggregate(IMDB.df$movie_title, by = list(IMDB.df$actor_3_name), FUN = length)
names(actor3count) <- c("Actor3Name", "MovieCount")
head(actor3count[order(actor3count$MovieCount, decreasing = TRUE), ], 30)

# dropping names and missing color
t(t(names(IMDB.df)))
IMDB.df <- IMDB.df[IMDB.df$color != "", -c(2, 7, 10:12, 15, 17:18)]
IMDB.df$color <- droplevels(IMDB.df$color)
dim(IMDB.df)
t(t(names(IMDB.df)))

# find frequencies for country
levels(IMDB.df$country)
sortcountry <- data.frame(table(IMDB.df$country))
sortcountry[order(sortcountry$Freq, decreasing = TRUE),]
# find countries that account for at least 1% of movies
bigcountry <- sortcountry[sortcountry$Freq / nrow(IMDB.df) > .01, ]
bigcountry
# keep these countries and group the rest into an "Other" category
IMDB.df$country2 <- as.factor(ifelse(IMDB.df$country %in% c("Australia","Canada","France","Germany",
                                                            "UK","USA"), 
                                     as.character(IMDB.df$country), "Other"))
IMDB.df[1:20, "country2"]
levels(IMDB.df$country2)

# language
# find frequencies for language
levels(IMDB.df$language)
sortlanguage <- data.frame(table(IMDB.df$language))
sortlanguage[order(sortlanguage$Freq, decreasing = TRUE), ]
# find languages that account for at least 1% of movies
biglanguage <- sortlanguage[sortlanguage$Freq / nrow(IMDB.df) > .01, ]
biglanguage
# keep these languages and group the rest into an "Other" category
IMDB.df$language2 <- as.factor(ifelse(IMDB.df$language %in% c("English","French"),
                                      as.character(IMDB.df$language), "Other"))
IMDB.df[1:20, "language2"]
levels(IMDB.df$language2)

# content rating
# grouping content_rating
levels(IMDB.df$content_rating)
boxplot(IMDB.df$imdb_score ~ IMDB.df$content_rating, las = 2)
sortrating <- data.frame(table(IMDB.df$content_rating))
sortrating[order(sortrating$Freq, decreasing = TRUE), ]
IMDB.df$rating2 <- factor(ifelse(substr(IMDB.df$content_rating, 1, 2) == "TV", "TV", 
                                 ifelse(IMDB.df$content_rating %in% c("R","PG-13","PG","Not Rated",
                                                                      "G"),
                                        as.character(IMDB.df$content_rating),"Other")))
levels(IMDB.df$rating2)
summary(IMDB.df$rating2)
summary(IMDB.df)

#face number in poster
data.for.plot <- aggregate(IMDB.df$imdb_score, by = list(IMDB.df$facenumber_in_poster), FUN = length)
names(data.for.plot) <- c("facenumber_in_poster", "CountOfMovies")
barplot(height = data.for.plot$CountOfMovies, names.arg = data.for.plot$facenumber_in_poster)
IMDB.df$facenumber_in_poster <- as.factor(ifelse(IMDB.df$facenumber_in_poster >= 7 | is.na(IMDB.df$facenumber_in_poster),
                                                 "MoreThan6",
                                                 as.character(IMDB.df$facenumber_in_poster)))

# aspect ratio #
# find frequencies for aspect_ratio
IMDB.df$aspect_ratio <- as.factor(IMDB.df$aspect_ratio)
levels(IMDB.df$aspect_ratio)
sortaspectratio <- data.frame(table(IMDB.df$aspect_ratio))
sortaspectratio[order(sortaspectratio$Freq, decreasing = TRUE), ]
# find aspect ratios that account for at least 1% of movies
bigaspectratio <- sortaspectratio[sortaspectratio$Freq / nrow(IMDB.df) > .01, ]
bigaspectratio
# keep these aspect ratio and group the rest into an "Other" category
IMDB.df$aspect_ratio2 <- as.factor(ifelse(IMDB.df$aspect_ratio %in% c("1.33","1.37","1.66","1.78","1.85","2.35"),
                                          as.character(IMDB.df$aspect_ratio), "Other"))
IMDB.df[1:20, "aspect_ratio2"]
levels(IMDB.df$aspect_ratio2)

str(IMDB.df)

# handling missing numeric data 
# deleting records
dim(IMDB.df[is.na(IMDB.df$num_critic_for_reviews) |
              is.na(IMDB.df$duration) |
              is.na(IMDB.df$actor_3_facebook_likes) |
              is.na(IMDB.df$actor_1_facebook_likes) |
              is.na(IMDB.df$num_user_for_reviews) |
              is.na(IMDB.df$actor_2_facebook_likes), ])
IMDB.df <- IMDB.df[!(is.na(IMDB.df$num_critic_for_reviews) |
                       is.na(IMDB.df$duration) |
                       is.na(IMDB.df$actor_3_facebook_likes) |
                       is.na(IMDB.df$actor_1_facebook_likes) |
                       is.na(IMDB.df$num_user_for_reviews) |
                       is.na(IMDB.df$actor_2_facebook_likes)), ]
dim(IMDB.df)

# imputing the median
IMDB.df$director_facebook_likes[is.na(IMDB.df$director_facebook_likes)] <- 
  median(IMDB.df$director_facebook_likes, na.rm = TRUE)
IMDB.df$gross[is.na(IMDB.df$gross)] <- median(IMDB.df$gross, na.rm = TRUE)
IMDB.df$title_year[is.na(IMDB.df$title_year)] <- median(IMDB.df$title_year, na.rm = TRUE)

summary(Filter(is.numeric, IMDB.df))

# drop budget due to data reliability issues
t(t(names(IMDB.df)))
IMDB.df <- IMDB.df[, -15]
summary(IMDB.df)


t(t(names(IMDB.df)))

# dummy variables
levels(IMDB.df$color)
levels(IMDB.df$country2)
levels(IMDB.df$language2)
levels(IMDB.df$rating2)
levels(IMDB.df$facenumber_in_poster)
levels(IMDB.df$aspect_ratio2)

library(fastDummies)
IMDB.df <- dummy_cols(IMDB.df,
                      select_columns = c("color", "country2","language2",
                                         "rating2","facenumber_in_poster",
                                         "aspect_ratio2"),
                      remove_selected_columns = TRUE)
t(t(names(IMDB.df)))

# drop the original categorical variables
IMDB.df <- IMDB.df[, -c(10:12, 16)]
t(t(names(IMDB.df)))

# partition the data
set.seed(11)
## partitioning into training (70%) and validation (30%) 
train.rows <- sample(rownames(IMDB.df), nrow(IMDB.df)*0.7)
IMDB.train <- IMDB.df[train.rows, ]
valid.rows <- setdiff(rownames(IMDB.df), train.rows)
IMDB.valid <- IMDB.df[valid.rows, ]

## normalize data
# initialize normalized training and validation data to originals
IMDB.train.norm <- IMDB.train
IMDB.valid.norm <- IMDB.valid

# normalize all predictors to a 0-1 scale
cols <- colnames(IMDB.train[, -12])
for (i in cols) {
  IMDB.valid.norm[[i]] <- 
    (IMDB.valid.norm[[i]] - min(IMDB.train[[i]])) / (max(IMDB.train[[i]]) - min(IMDB.train[[i]]))
  IMDB.train.norm[[i]] <- 
    (IMDB.train.norm[[i]] - min(IMDB.train[[i]])) / (max(IMDB.train[[i]]) - min(IMDB.train[[i]]))
}
summary(IMDB.train.norm)
summary(IMDB.valid.norm)
t(t(names(IMDB.train.norm)))

# stepwise regression
IMDB.lm <- lm(IMDB.train$imdb_score ~ ., data = IMDB.train)         
IMDB.lm.null <- lm(IMDB.train$imdb_score ~ 1, data = IMDB.train) 

IMDB.lm.step <- step(IMDB.lm.null,                                    
                     scope = list(IMDB.lm.null, upper = IMDB.lm),   
                     direction = "both")                                

summary(IMDB.lm.step)

# k-NN regression with k=1
library(FNN)
library(caret)
IMDB.nn <- knn.reg(train = IMDB.train.norm[, c(1:2,4,7,9:10,13,15,34,36:37,40:41,43:44,
                                               46:47,51,53,56,75,98,100:103,105,107,117)],
                   test = IMDB.valid.norm[, c(1:2,4,7,9:10,13,15,34,36:37,40:41,43:44,
                                              46:47,51,53,56,75,98,100:103,105,107,117)], 
                   y = IMDB.train.norm$imdb_score,
                   k = 1)

# compile the actual and predicted values and view the first 20 records
IMDB.nn.results <- data.frame(cbind(pred = IMDB.nn$pred, actual = IMDB.valid.norm$imdb_score))
head(IMDB.nn.results, 20)
RMSE(IMDB.nn$pred, 
     IMDB.valid.norm$imdb_score)

# finding optimal k using RMSE
# initialize a data frame with two columns: k and accuracy
RMSE.df <- data.frame(k = seq(1, 30, 1), RMSE.k = rep(0, 30))

# compute knn for different k on validation set
for (i in 1:30) {
  knn.reg.pred <- knn.reg(train = IMDB.train.norm[, c(1:2,4,7,9:10,13,15,34,36:37,40:41,43:44,
                                                      46:47,51,53,56,75,98,100:103,105,107,117)], 
                          test = IMDB.valid.norm[, c(1:2,4,7,9:10,13,15,34,36:37,40:41,43:44,
                                                     46:47,51,53,56,75,98,100:103,105,107,117)], 
                          y = IMDB.train.norm$imdb_score, 
                          k = i)
  RMSE.df[i, 2] <- RMSE(IMDB.valid.norm$imdb_score, 
                        knn.reg.pred$pred)
}
RMSE.df

# k-NN with optimal k
IMDB.nn.best <- knn.reg(train = IMDB.train.norm[, c(1:2,4,7,9:10,13,15,34,36:37,40:41,43:44,
                                                    46:47,51,53,56,75,98,100:103,105,107,117)], 
                        test = IMDB.valid.norm[, c(1:2,4,7,9:10,13,15,34,36:37,40:41,43:44,
                                                   46:47,51,53,56,75,98,100:103,105,107,117)], 
                        y = IMDB.train.norm$imdb_score, 
                        k = 11)
RMSE(IMDB.nn.best$pred, 
     IMDB.valid.norm$imdb_score)
