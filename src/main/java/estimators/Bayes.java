package estimators;

import org.javatuples.Pair;
import org.la4j.Matrix;
import org.la4j.Vector;

import static sun.java2d.xr.XRUtils.None;


public class Bayes {
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
    Matrix trainX;
    Vector trainY;
    Matrix testX;
    Vector testY;

    public Bayes(double[][] trainX, double[] trainY, double[][] testX, double[] testY) {
        this.testX = Matrix.from2DArray(testX);
        this.trainX = Matrix.from2DArray(trainX);
        this.testY = Vector.fromArray(testY);
        this.trainY = Vector.fromArray(trainY);
    }

    public Bayes(double[] trainX, double[] trainY, double[] testX, double[] testY) {
        this.testX = Matrix.from1DArray(testX.length, 1, testX);
        this.trainX = Matrix.from1DArray(trainX.length, 1, trainX);
        this.testY = Vector.fromArray(testY);
        this.trainY = Vector.fromArray(trainY);
    }

    public Bayes(
            int n_iter,
            double tol,
            double alpha_1,
            double alpha_2,
            double lambda_1,
            double lambda_2,
            double alpha_init,
            double lambda_init,
            boolean compute_score,
            boolean fit_intercept,
            boolean copy_X,
            boolean verbose) {
        this.n_iter = n_iter;
        this.tol = tol;
        this.alpha_1 = alpha_1;
        this.alpha_2 = alpha_2;
        this.lambda_1 = lambda_1;
        this.lambda_2 = lambda_2;
        this.alpha_init = alpha_init;
        this.lambda_init = lambda_init;
        this.compute_score = compute_score;
        this.fit_intercept = fit_intercept;
        this.copy_X = copy_X;
        this.verbose = verbose;
    }
/*    """Bayesian ridge regression.

    Fit a Bayesian ridge model. See the Notes section for details on this
    implementation and the optimization of the regularization parameters
    lambda (precision of the weights) and alpha (precision of the noise).

    Read more in the :ref:`User Guide <bayesian_regression>`.

    Parameters
    ----------
    n_iter : int, default=300
        Maximum number of iterations. Should be greater than or equal to 1.

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    alpha_init : float, default=None
        Initial value for alpha (precision of the noise).
        If not set, alpha_init is 1/Var(y).

            .. versionadded:: 0.22

    lambda_init : float, default=None
        Initial value for lambda (precision of the weights).
        If not set, lambda_init is 1.

            .. versionadded:: 0.22

    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration of the
        optimization.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        The intercept is not treated as a probabilistic parameter
        and thus has no associated variance. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    alpha_ : float
       Estimated precision of the noise.

    lambda_ : float
       Estimated precision of the weights.

    sigma_ : array-like of shape (n_features, n_features)
        Estimated variance-covariance matrix of the weights

    scores_ : array-like of shape (n_iter_+1,)
        If computed_score is True, value of the log marginal likelihood (to be
        maximized) at each iteration of the optimization. The array starts
        with the value of the log marginal likelihood obtained for the initial
        values of alpha and lambda and ends with the value obtained for the
        estimated alpha and lambda.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    X_offset_ : ndarray of shape (n_features,)
        If `fit_intercept=True`, offset subtracted for centering data to a
        zero mean. Set to np.zeros(n_features) otherwise.

    X_scale_ : ndarray of shape (n_features,)
        Set to np.ones(n_features).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    ARDRegression : Bayesian ARD regression.

    Notes
    -----
    There exist several strategies to perform Bayesian ridge regression. This
    implementation is based on the algorithm described in Appendix A of
    (Tipping, 2001) where updates of the regularization parameters are done as
    suggested in (MacKay, 1992). Note that according to A New
    View of Automatic Relevance Determination (Wipf and Nagarajan, 2008) these
    update rules do not guarantee that the marginal likelihood is increasing
    between two consecutive iterations of the optimization.

    References
    ----------
    D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems,
    Vol. 4, No. 3, 1992.

    M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine,
    Journal of Machine Learning Research, Vol. 1, 2001.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.BayesianRidge()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    BayesianRidge()
    >>> clf.predict([[1, 1]])
    array([1.])
    """*/

    /*_parameter_constraints: dict = {
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "alpha_init": [None, Interval(Real, 0, None, closed="left")],
        "lambda_init": [None, Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }*/


    public Bayes fit(double[][] X, double[] y) {
        /*"""Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.20
               parameter *sample_weight* support to BayesianRidge.

        Returns
        -------
        self : object
            Returns the instance itself.
        """*/
        this._validate_params();

        //X, y = this._validate_data(X, y, dtype=[np.float64, np.float32], y_numeric=True);

//        if (sample_weight != null){
//            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype);
//        }

        X, y, X_offset_, y_offset_, X_scale_ = this._preprocess_data(
                X,
                y,
                this.fit_intercept,
                this.copy_X
                //sample_weight=sample_weight,
        );

//        if (sample_weight !=null) {
//            //Sample weight can be implemented via a simple rescaling.;
//            X, y, _ = _rescale_data(X, y, sample_weight);
//        }

        this.X_offset_ = X_offset_;
        this.X_scale_ = X_scale_;
        n_samples, n_features = X.shape;

        // Initialization of the values of the parameters;
        eps = np.finfo(np.float64).eps;
        // Add `eps` in the denominator to omit division by zero if `np.var(y)`;
        // is zero;
        alpha_ = this.alpha_init;
        lambda_ = this.lambda_init;
        if alpha_ is None:
        ;
        alpha_ = 1.0 / (np.var(y) + eps);
        if lambda_ is None:
        ;
        lambda_ = 1.0;

        verbose = this.verbose;
        lambda_1 = this.lambda_1;
        lambda_2 = this.lambda_2;
        alpha_1 = this.alpha_1;
        alpha_2 = this.alpha_2;

        this.scores_ = list();
        coef_old_ = None;

        XT_y = np.dot(X.T, y);
        U, S, Vh = linalg.svd(X, full_matrices = False);
        eigen_vals_ = S **2;

        // Convergence loop of the bayesian ridge regression;
        for iter_ in range(this.n_iter):;

        // update posterior mean coef_ based on alpha_ and lambda_ and;
        // compute corresponding rmse;
        coef_, rmse_ = this._update_coef_(;
        X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_;
        );
        if this.compute_score:;
        // compute the log marginal likelihood;
        s = this._log_marginal_likelihood(;
        n_samples, n_features, eigen_vals_, alpha_, lambda_, coef_, rmse_;
        );
        this.scores_.append(s);

        // Update alpha and lambda according to (MacKay, 1992);
        gamma_ = np.sum((alpha_ * eigen_vals_) / (lambda_ + alpha_ * eigen_vals_));
        lambda_ = (gamma_ + 2 * lambda_1) / (np.sum(coef_ * * 2) + 2 * lambda_2);
        alpha_ = (n_samples - gamma_ + 2 * alpha_1) / (rmse_ + 2 * alpha_2);

        // Check for convergence;
        if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < this.tol:;
        if verbose:
        ;
        print("Convergence after ", str(iter_), " iterations");
        break;
        coef_old_ = np.copy(coef_);

        this.n_iter_ = iter_ + 1;

        // return regularization parameters and corresponding posterior mean,;
        // log marginal likelihood and posterior covariance;
        this.alpha_ = alpha_;
        this.lambda_ = lambda_;
        this.coef_, rmse_ = this._update_coef_(;
        X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_;
        );
        if this.compute_score:;
        // compute the log marginal likelihood;
        s = this._log_marginal_likelihood(;
        n_samples, n_features, eigen_vals_, alpha_, lambda_, coef_, rmse_;
        );
        this.scores_.append(s);
        this.scores_ = np.array(this.scores_);

        // posterior covariance is given by 1/alpha_ * scaled_sigma_;
        scaled_sigma_ = np.dot(;
        Vh.T, Vh / (eigen_vals_ + lambda_ / alpha_)[:,np.newaxis];
        );
        this.sigma_ = (1.0 / alpha_) * scaled_sigma_;

        this._set_intercept(X_offset_, y_offset_, X_scale_);

        return this;
    }

    def predict(self, X, return_std=False):
            """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array-like of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
    y_mean =self._decision_function(X)
            if
    not return_std:
            return y_mean
        else:
    sigmas_squared_data =(np.dot(X,self.sigma_)*X).

    sum(axis=1)

    y_std =np.sqrt(sigmas_squared_data +(1.0/self.alpha_))
            return y_mean,y_std

    private Pair _update_coef_(
            Matrix X, Vector y, int n_samples, int n_features, Matrix XT_y, Matrix U, Matrix Vh, Vector eigen_vals_, double alpha_, double lambda_
    ) {
        /*"""Update posterior mean and compute corresponding rmse.

        Posterior mean is given by coef_ = scaled_sigma_ * X.T * y where
        scaled_sigma_ = (lambda_/alpha_ * np.eye(n_features)
                         + np.dot(X.T, X))^-1
        """*/
        Matrix coef_;
        Vector broadcast_vec = eigen_vals_.add(lambda_ / alpha_);
        broadcast_vec = broadcast_vec.apply(new LinOps.InvertV<Vector>());
        if (n_samples > n_features) {
            Matrix broadcast_mat = Matrix.zero(Vh.rows(), Vh.columns());
            for (int i = 0; i < Vh.columns(); i++) {
                broadcast_mat.setColumn(i, broadcast_vec);
            }
            coef_ = Vh.transpose().multiply(Vh.hadamardProduct(broadcast_mat)).multiply(XT_y);
            /*coef_ =np.linalg.multi_dot(
                [Vh.transpose(),
                    Vh /(eigen_vals_ +lambda_ /alpha_)[:,np.newaxis],
                    XT_y]
            )*/
        } else {
            Matrix broadcast_y = Matrix.zero(X.rows(), X.columns());
            for (int i = 0; i < X.columns(); i++) {
                broadcast_y.setColumn(i, y);
            }
            Matrix broadcast_mat = Matrix.zero(U.rows(), U.columns());
            for (int i = 0; i < Vh.columns(); i++) {
                broadcast_mat.setColumn(i, broadcast_vec);
            }
            coef_ = X.transpose().multiply(U.hadamardProduct(broadcast_mat)).multiply(U.transpose()).multiply(broadcast_y);
            /*coef_ = np.linalg.multi_dot(
                [X.transpose(),
                    U / (eigen_vals_ + lambda_ / alpha_)[None,:],
                    U.transpose(),
                    y]
            )*/
        }

        double rmse_ = np.sum((y - np.dot(X, coef_)) * * 2)
        Matrix a = X.multiply(coef_);

        return new Pair(coef_, rmse_);
    }

    def _log_marginal_likelihood(
            self, n_samples, n_features, eigen_vals, alpha_, lambda_, coef, rmse
            ):
            """Log marginal likelihood."""
    alpha_1 =
    self.alpha_1
            alpha_2 = self.alpha_2
    lambda_1 =
    self.lambda_1
            lambda_2 = self.lambda_2

        #
    compute the
    log of
    the determinant
    of the
    posterior covariance.
            #
    posterior covariance
    is given
    by
        #sigma =(lambda_ *np.eye(n_features)+alpha_ *np.dot(X.T,X))^-1
            if n_samples >n_features:
    logdet_sigma =-np.sum(np.log(lambda_ +alpha_ *eigen_vals))
            else:
    logdet_sigma =np.full(n_features,lambda_,dtype=np.array(lambda_).dtype)
    logdet_sigma[:n_samples]+=alpha_ *
    eigen_vals
            logdet_sigma = -np.sum(np.log(logdet_sigma))

    score =lambda_1 *

    log(lambda_) -lambda_2 *lambda_
    score +=alpha_1 *

    log(alpha_) -alpha_2 *alpha_
    score +=0.5*(
    n_features *

    log(lambda_)
            +n_samples *

    log(alpha_)
            -alpha_ *rmse
            -lambda_ *np.sum(coef**2)
            +logdet_sigma
            -n_samples *

    log(2*np.pi)
        )

                return score
}
