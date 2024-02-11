
data {
    int<lower=0> N_years;
    int<lower=0> N_weeks;
    int<lower=0> N_stations;
    int<lower=0> N_covariates;

    int<lower=0> N_years_to_pred;
    int<lower=0> N_weeks_to_pred;
    int<lower=0> N_stations_to_pred;

    array[N_years,N_weeks,N_stations] real y;
    array[N_years,N_weeks] matrix[N_stations,N_covariates] delta;
    matrix[N_stations,N_stations] dist; //distance matrix

    matrix[N_stations_to_pred, N_stations] dist_pred; // distance matrix which represents the distance from the stations to predict to every other station in the data set
    matrix[N_stations_to_pred, N_stations_to_pred] dist_pred_to_pred; // distance matrix which represents the distance between the stations to predict
    vector[N_weeks_to_pred] weeks_to_pred; // represents the time on which we want to perform prediction
    array[N_years_to_pred,to_int(max(weeks_to_pred))] matrix[N_stations,N_covariates] delta_pred_sim_t; // represents the array of delta value in the weeks on which we want to do the prediction in the observed location
    array[N_years_to_pred,to_int(max(weeks_to_pred))] matrix[N_stations_to_pred,N_covariates] delta_pred_sim_s_t; // represents the array of delta value in the weeks and sensors on which we want to do the prdiction

    real<lower=0> a;
    real<lower=0> b;
    real<lower=0> s0;
    real<lower=0> phi_gamma_esti;
    real<lower=0> phi_eta_esti;
}

parameters {
    real<lower=0,upper=1> rho;
    array[N_years] real xi;
    array[N_covariates] real beta;
    array[N_years,N_weeks,N_stations] real eta;
    array[N_years] real mu;
    array[N_years,N_stations] real gamma;

    real<lower=0> tau_eta;
    real<lower=0> tau_gamma;
    real<lower=0> tau_epsilon;
}

transformed parameters {
    matrix[N_stations,N_stations] Sigma_eta;
    matrix[N_stations,N_stations] Sigma_gamma;
    for (i in 1:N_stations){
        for (j in 1:N_stations){
            Sigma_eta[i,j]=(1/tau_eta)*exp(-phi_eta_esti*dist[i,j]);
            Sigma_gamma[i,j]=(1/tau_gamma)*exp(-phi_gamma_esti*dist[i,j]);
        }
    }
}

model {
    vector[N_stations] zero = rep_vector(0,N_stations);
    rho ~ normal(0,1);
    xi ~ normal(0, s0);//check if a for loop is needed
    beta ~ normal(0, s0);//same
    tau_eta ~ gamma(a, b);
    tau_gamma ~ gamma(a, b);
    tau_epsilon ~ gamma(a, b);
    mu ~ normal(0,s0);//same

    for (i in 1:N_years){
      for (j in 1:N_weeks){
        to_vector(eta[i][j]) ~ multi_normal(zero,Sigma_eta);
      }
    }

    for (i in 1:N_years){
      to_vector(gamma[i]) ~ multi_normal(zero,Sigma_gamma);
    }

    for (i in 1:N_years) {
      y[i][1] ~ normal(rep_vector(mu[i],N_stations)+to_vector(gamma[i]), 1/tau_epsilon);
      for (j in 2:N_weeks){
        y[i][j] ~ normal(rho*to_vector(y[i][j-1])+rep_vector(xi[i],N_stations)+delta[i][j]*to_vector(beta)+to_vector(eta[i][j]),1/tau_epsilon);//Check matrix product
      }
    }
}

generated quantities {
    vector[N_stations] zero = rep_vector(0,N_stations);
    // Posterior predictive distribution
    array[N_years_to_pred, N_stations_to_pred] real gamma_pred_sim; // represents values of gamma in a new location
    array[N_years_to_pred,to_int(max(weeks_to_pred)),N_stations] real o_pred_sim_t; // represents the O value predicted for new time
    array[N_years_to_pred,to_int(max(weeks_to_pred)),N_stations_to_pred] real o_pred_sim_s_t; // represents the O value predicted for a new location
    array[N_years_to_pred,N_weeks_to_pred,N_stations_to_pred] real y_pred_sim;// represents the array of values for ozon level predicted for new time and new location

    // Calculate the derived quantity Sigma_gamma_12
    matrix[N_stations_to_pred, N_stations] Sigma_gamma_12;
    for (i in 1:N_stations_to_pred){
      for (j in 1:N_stations){
        Sigma_gamma_12[i,j] = exp(-phi_gamma_esti * dist_pred[i][j]);
      }
    }

    // Claculate the derived quantity Sigma_eta_12
    matrix[N_stations_to_pred, N_stations] Sigma_eta_12;
    for (i in 1:N_stations_to_pred){
      for (j in 1:N_stations){
        Sigma_eta_12[i,j] = exp(-phi_eta_esti * dist_pred[i][j]);
      }
    }

    // Inverse of Sigma_gamma
    matrix[N_stations, N_stations] Sigma_gamma_inv;
    Sigma_gamma_inv = inverse(Sigma_gamma);

    // Inverse of Sigma_eta
    matrix[N_stations, N_stations] Sigma_eta_inv;
    Sigma_eta_inv = inverse(Sigma_eta);

    // Draw gamma_pred_sim from its posterior distribution
    for (i in 1:N_years_to_pred){
      for (j in 1:N_stations_to_pred){
        gamma_pred_sim[i,j] = normal_rng(Sigma_gamma_12[i, ] * Sigma_gamma_inv * to_vector(gamma[i]), (1/tau_gamma) * (1 - Sigma_gamma_12[i, ] * Sigma_gamma_inv * Sigma_gamma_12[i, ]'));
      }
    }

    // Draw o_pred_sim_t from its posterior predictive distribution
    for (i in 1:N_years_to_pred){
      o_pred_sim_t[i][1] = to_array_1d(to_vector(gamma[i]) + mu[i]);
      for (j in 2:to_int(max(weeks_to_pred))){
        o_pred_sim_t[i][j] = to_array_1d(multi_normal_rng(xi[i] + rho *  to_vector(o_pred_sim_t[i][j - 1]) + delta_pred_sim_t[i][j] * to_vector(beta), (1/tau_eta) * Sigma_eta));
      }
    }
    //Draw o_pred_sim_s_t from its posterior predictive distribution
    for (i in 1:N_years_to_pred){
      for (j in 1:N_stations_to_pred){
        o_pred_sim_s_t[i][1][j] = gamma_pred_sim[i,j] + mu[i];
        for (t in 2:to_int(max(weeks_to_pred))){
          real G_delta = (1/tau_eta) * (1 - Sigma_eta_12[j, ] * Sigma_eta_inv * Sigma_eta_12[j, ]');
          real arg_1 = xi[i] + rho * o_pred_sim_s_t[i][t-1][j] + delta_pred_sim_s_t[i][t][j] * to_vector(beta);
          real arg_2 = Sigma_eta_12[j, ] * Sigma_eta_inv * (to_vector(o_pred_sim_t[i][j]) - xi[i] - rho *  to_vector(o_pred_sim_t[i][t - 1]) - delta_pred_sim_t[i][j] * to_vector(beta));
          real zeta = arg_1 + arg_2;
          o_pred_sim_s_t[i][t][j] = normal_rng(zeta, abs(G_delta));
        }
      }
    }

    // Draw the Ozon Level in the new location and new time from its posterior predictive distribution
    for (i in 1:N_years_to_pred){
      for (j in 1:N_weeks_to_pred){
        for (k in 1:N_stations_to_pred){
          y_pred_sim[i][j][k] = normal_rng(o_pred_sim_s_t[i][to_int(weeks_to_pred[j])][k], 1/tau_epsilon);
        }
      }
    }
}

