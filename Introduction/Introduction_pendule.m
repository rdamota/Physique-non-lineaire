%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                                                  %%%%%
%%%%%     Physique non-linéaire: le pendule pesant     %%%%%
%%%%%                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% L'équation du pendule pesant avec amortissement est donné par:
%     d2x/dt2 + lambda dx/dt + omega0^2 x = 0

close all;
clear all;

%---------------------------------------------
%----- Pendule pesant sans amortissement -----
%---------------------------------------------

%--> Définition du système que l'on cherche à simuler.
pendule = @(t, y) [y(2) ; -sin(y(1))];

%--> Création du maillage pour la partie du plan de phase qui nous intéresse.
u = linspace(-2*pi, 2*pi, 25);
v = linspace(-pi, pi, 25);

[u, v] = meshgrid(u, v);

%--> Calcul de du/dt et dv/dt sur chacun des points du maillage.
udot = zeros(size(u)); vdot = zeros(size(v));
for i = 1:numel(u)
  yp = pendule(0, [u(i), v(i)]);
  udot(i) = yp(1);
  vdot(i) = yp(2);
end

%--> Plot le plan de phase du système.
fig = figure(1);
quiver(u, v, udot, vdot, 'k');
xlabel('u'); ylabel('v'); axis equal tight;
title('Plan de phase du pendule pesant sans amortissement')
hold on;

%--> Ajout des points d'équilibre.
eq_u = -2*pi:pi:2*pi; eq_v = zeros(size(eq_u));
plot(eq_u, eq_v, 'r.', 'MarkerSize', 12);

%--> Simulation et plot de différentes trajectoirs.
t = linspace(0, 50, 1000);
for v_init = 0:0.5:2.5
  %--> Condition initiale.
  y0 = [0; v_init];
  %--> Intégration temporelle.
  [ts, y] = ode45(pendule, t, y0);
  %--> Plot la trajectoire.
  plot(y(:, 1), y(:, 2), 'LineWidth', 2.5);
end


%---------------------------------------------
%----- Pendule pesant avec amortissement -----
%---------------------------------------------

prompt = 'Quelle valeur du coefficient d amortissement souhaitez-vous utiliser?   ';
lambda = input(prompt);

%--> Définition du système que l'on cherche à simuler.
pendule_bis = @(t, y) [y(2) ; -lambda*y(2)-sin(y(1))];

%--> Calcul de du/dt et dv/dt sur chacun des points du maillage.
udot = zeros(size(u)); vdot = zeros(size(v));
for i = 1:numel(u)
  yp = pendule_bis(0, [u(i), v(i)]);
  udot(i) = yp(1);
  vdot(i) = yp(2);
end

%--> Plot le plan de phase du système.
fig = figure(2);
quiver(u, v, udot, vdot, 'k');
xlabel('u'); ylabel('v'); axis equal tight;
title('Plan de phase du pendule pesant avec amortissement')
hold on;

%--> Ajout des points d'équilibre.
eq_u = -2*pi:pi:2*pi; eq_v = zeros(size(eq_u));
plot(eq_u, eq_v, 'r.', 'MarkerSize', 12);

%--> Simulation et plot de différentes trajectoirs.
t = linspace(0, 50, 1000);
for v_init = 0:0.5:2.5
  %--> Condition initiale.
  y0 = [0; v_init];
  %--> Intégration temporelle.
  [ts, y] = ode45(pendule_bis, t, y0);
  %--> Plot la trajectoire.
  plot(y(:, 1), y(:, 2), 'LineWidth', 2.5);
end
