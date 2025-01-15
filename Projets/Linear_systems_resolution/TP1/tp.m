close all;
clear all;

%name = 'mat1';
name = 'pde225_5e-1';
%name = 'hydcar20.mat';

load(name);

n = size(A,1);
fprintf('dimension de A : %4d \n', n);

b = [1:n]';

x0 = zeros(n, 1);

eps = 1e-6;

kmax = n;

% FOM
fprintf('FOM\n');
[x, flag, relres, iter, resvec] = krylov(A, b, x0, eps, kmax, 0);
fprintf('Nb iterations : %4d \n', iter);
semilogy(resvec/norm(b), 'c'); 
if(flag ~= 0)
    fprintf('pas de convergence\n');
end

% Ajout du titre et du nom des axes sur la figure
title(sprintf('Convergence de Krylov, GMRES et GMRES MATLAB, epsilon = %g', eps)); 
xlabel('Nombre d''itérations'); 
ylabel('Norme relative du résidu'); 


% Précision de l'algorithme
fprintf("Précision de l'algorithme FOM = %3d\n",relres)

pause

% GMRES
fprintf('GMRES\n');
[x, flag, relres, iter, resvec] = krylov(A, b, x0, eps, kmax, 1);
fprintf('Nb iterations : %4d \n', iter);
hold on
semilogy(resvec/norm(b), 'r'); 
if(flag ~= 0)
    fprintf('pas de convergence\n');
end

% Précision de l'algorithme
fprintf("Précision de l'algorithme GMRES implémenté = %3d\n",relres)

pause

% GMRES MATLAB
fprintf('GMRES MATLAB\n');
[x, flag, relres, iter, resvec] = gmres(A, b, [], eps, kmax, [], [], x0);
fprintf('Nb iterations : %4d \n', iter);
hold on
semilogy(resvec/norm(b), '+');
if(flag ~= 0)
    fprintf('pas de convergence\n');
end

% Ajout de la légende sur la figure
legend('FOM', 'GMRES', 'GMRES MATLAB'); 

% Précision de l'algorithme
fprintf("Précision de l'algorithme GMRES de Matlab= %3d\n",relres)

pause

% Visualisation de la matrice A
figure;
spy(A); 
title(sprintf('Matrice A = %s', name)); 
