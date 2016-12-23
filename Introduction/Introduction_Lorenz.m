close all;
clear all;

prompt = 'Pour quelle valeur de rho souhaitez-vous simuler le système?    ';
rho = input(prompt);

%--> Définition des paramètres du système.
sigma = 10; beta = 8/3;

%--> Définition du système dynamique.
f = @(t, y) [sigma*(y(2)-y(1)) ; y(1)*(rho - y(3))-y(2) ; y(1)*y(2) - beta*y(3)];

%--> Points d'équilibre du système.
eq_x = [0, sqrt(beta*(rho-1)), -sqrt(beta*(rho-1))];
eq_y = [0, sqrt(beta*(rho-1)), -sqrt(beta*(rho-1))];
eq_z = [0, rho-1, rho-1];

%--> Définition de la condition initiale.
y0_a = [0.0001, 0.0001, 0.0001];
y0_b = [0.0001, -0.0001, 0.0001];

%--> Simulation du système.
t = linspace(0, 100, 20000);
[ts, ya] = ode45(f, t, y0_a);
[ts, yb] = ode45(f, t, y0_b);

%--> Trace l'évolution temporelle des différentes composantes du système.
figure(1);
subplot(3, 1, 1); plot(ts, ya(:, 1), ts, yb(:, 1)); ylabel('x'); set(gca, 'XTickLabel', "");
subplot(3, 1, 2); plot(ts, ya(:, 2), ts, yb(:, 2)); ylabel('y');  set(gca, 'XTickLabel', "");
subplot(3, 1, 3); plot(ts, ya(:, 3), ts, yb(:, 3)); ylabel('z'); xlabel('t');

%--> Trajectoire dans l'espace des phases.
figure(2);
plot3(ya(:, 1), ya(:, 2), ya(:, 3)); hold on; plot3(yb(:, 1), yb(:, 2), yb(:, 3), '--');
plot3(eq_x, eq_y, eq_z, 'r.', 'MarkerSize', 12); xlabel('x'); ylabel('y'), zlabel('z');

%--> Change the view.
view(45, 30)
