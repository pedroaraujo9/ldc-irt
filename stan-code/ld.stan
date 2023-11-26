data{
    int T; // number of tests
    int n; // number of respondents
    int J; // number of items
    int X[n, J, T]; // binary items
}


parameters{
    matrix[J, T] b; // difficulty
    matrix<lower=0>[J, T] a; // discrimination
    vector[2] Z[T]; // latent coordinate
    real<lower=0> phi; // scale
    vector[T] eta[n]; // centralized latent trait
}

transformed parameters {
    corr_matrix[T] corr_theta; // intertrait correlation
    vector[T] theta[n]; // latent trait
    matrix[T, T] L; // cholesky decomposition of corr_theta

    // Latent distance correlation structure
    for(l in 1:T) {
        for(k in l:T) {
            if(l == k) {
                corr_theta[l,k] = 1;
            }else{
                corr_theta[l,k] = exp(-distance(Z[l], Z[k])/phi);
                corr_theta[k,l] = exp(-distance(Z[l], Z[k])/phi);
            }
        }
    }

    L = cholesky_decompose(corr_theta);

    // adjusting latent traits
    for(i in 1:n) {
        theta[i,] = L * eta[i];
    }
}

model{
    // prior for the scale parameter
    phi ~ cauchy(0, 10);

    // prior for the latent coordinates
    for(t in 1:T) {
        Z[t] ~ std_normal();
    }

    // prior for non-centralized latent traits
    for(i in 1:n) {
        eta[i] ~ std_normal();
    }

    // prior for difficulty and discrimination
    for(j in 1:J) {
        b[j, ] ~ std_normal();
        a[j, ] ~ gamma(0.1, 0.1);
    }

    // likelihood
    for(i in 1:n) {
        for(t in 1:T) {
            for(j in 1:J) {
                X[i, j, t] ~ bernoulli_logit(a[j, t]*(theta[i, t] - b[j, t]));
            }
        }
    }
}