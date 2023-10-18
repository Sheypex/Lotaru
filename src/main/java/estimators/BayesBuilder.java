package estimators;

public class BayesBuilder {
    int n_iter;
    double tol;
    double alpha_1;
    double alpha_2;
    double lambda_1;
    double lambda_2;
    double alpha_init;
    double lambda_init;
    boolean compute_score;
    boolean fit_intercept;
    boolean copy_X;
    boolean verbose;

    public BayesBuilder() {
        this.n_iter = 300;
        this.tol = 1.0e-3;
        this.alpha_1 = 1.0e-6;
        this.alpha_2 = 1.0e-6;
        this.lambda_1 = 1.0e-6;
        this.lambda_2 = 1.0e-6;
        this.alpha_init = 0;
        this.lambda_init = 1;
        this.compute_score = false;
        this.fit_intercept = true;
        this.copy_X = true;
        this.verbose = false;
    }

    public Bayes makeBayes() {
        return new Bayes(n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, alpha_init, lambda_init, compute_score, fit_intercept, copy_X, verbose);
    }

    public BayesBuilder niter(int niter) {
        this.n_iter = niter;
        return this;
    }
}
