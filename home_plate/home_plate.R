setwd(here::here())

require(tidyverse)

df <- read.csv("home_plate/all_data.csv")
# distance to net
# 0/1 for best and exp
# puck location = event location
net <- read.csv("home_plate/net_coords.csv") %>%
  dplyr::mutate(X = round(X),
                Y = round(Y))
home_plate <- read.csv("home_plate/home_plate_coords.csv") %>%
  dplyr::mutate(X = round(X) + 100,
                Y = round(Y) + 85/2)

# plot(home_plate$X, home_plate$Y)

df <- df %>%
  dplyr::mutate(
    best_distance_to_net = sqrt((Max_Player_Best_X - net$X)^2 + (Max_Player_Best_Y - net$Y)^2),
    max_distance_to_net = sqrt((Max_Player_Exp_X - net$X)^2 + (Max_Player_Exp_Y - net$Y)^2),
    best_in_box = ifelse(Max_Player_Best_X >= 30 & Max_Player_Best_X <= 46 &
                          Max_Player_Best_Y >= 19.5 & Max_Player_Best_Y <= 64.5, 1, 0),
    best_in_skinny = ifelse(Max_Player_Best_X >= 11 & Max_Player_Best_X <= 30 &
                              Max_Player_Best_Y >= 39.5 & Max_Player_Best_Y <= 46.5, 1, 0),
    best_in_lower_triangle = ifelse(Max_Player_Best_Y >= 19.5 & Max_Player_Best_Y <= 39.5 & 
                                      Max_Player_Best_Y >= -1 * Max_Player_Best_X + 50.5 &
                                      Max_Player_Best_X <= 46, 1, 0),
    best_in_upper_triangle = ifelse(Max_Player_Best_Y >= 46.5 & Max_Player_Best_Y <= 64.5 & 
                                      Max_Player_Best_Y <= 1 * Max_Player_Best_X + 35.5 &
                                      Max_Player_Best_X <= 46, 1, 0),
    best_in_plate = pmax(best_in_box, best_in_skinny, best_in_lower_triangle, best_in_upper_triangle),
    
    max_in_box = ifelse(Max_Player_Exp_X >= 30 & Max_Player_Exp_X <= 46 &
                           Max_Player_Exp_Y >= 19.5 & Max_Player_Exp_Y <= 64.5, 1, 0),
    max_in_skinny = ifelse(Max_Player_Exp_X >= 11 & Max_Player_Exp_X <= 30 &
                              Max_Player_Exp_Y >= 39.5 & Max_Player_Exp_Y <= 46.5, 1, 0),
    max_in_lower_triangle = ifelse(Max_Player_Exp_Y >= 19.5 & Max_Player_Exp_Y <= 39.5 & 
                                      Max_Player_Exp_Y >= -1 * Max_Player_Exp_X + 50.5 &
                                      Max_Player_Exp_X <= 46, 1, 0),
    max_in_upper_triangle = ifelse(Max_Player_Exp_Y >= 46.5 & Max_Player_Exp_Y <= 64.5 & 
                                      Max_Player_Exp_Y <= 1 * Max_Player_Exp_X + 35.5 &
                                      Max_Player_Exp_X <= 46, 1, 0),
    max_in_plate = pmax(max_in_box, max_in_skinny, max_in_lower_triangle, max_in_upper_triangle)
  ) 

df %>%
  ggplot() +
  # geom_point(aes(x = Max_Player_Best_X, y = Max_Player_Best_Y, color = best_in_plate)) +
  geom_point(aes(x = Max_Player_Exp_X, y = Max_Player_Exp_Y, color = max_in_plate)) +
  geom_line(data = home_plate, aes(x = X, y = Y))

write.csv(df, file = "df_in_plate.csv", row.names = FALSE)
