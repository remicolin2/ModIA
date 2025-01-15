% Les 3 matrices ci dessous sont non symétriques 
% 
%load piston.mat;
%load pde225_5e-1.mat;
load hydcar20.mat;


n = size(A, 1);

b = [1:n]';

% resolution du systeme Ax = b
x_sol = A\b;

% Factorisation LU
[L, U, Pn] = lu(A);

spy(L)

fprintf("Nombre de non zéros de A : %4d \n", 2*nnz(A));
fprintf("Nombre d'opérations pour la phase de résolution : %4d \n", 2*nnz(L) - 2*n + 2*nnz(U) - n);


y = L\Pn*b;
x1 = U\y;

norm(b-A*x1)/norm(b);
norm(x_sol - x1) / norm(x_sol);


% Permutation symétrique symamd
%P = symamd(A);
%P = symrcm(A);
%P = amd(A);
%P = colamd(A);
P = colperm(A);


B = A(P,P);
%B = A(P, :);
B = A(:,P);

[L, U, Pn] = lu(B);

y = L\Pn*b(P);
%y = L\Pn*b;
x2 = U\y;
%x2(P) = x2;


% Nombre d'opérations et précision
%Erreur inverse normwise
normwise_res = norm(b-A*x2)/norm(b); 
normwise_sol = norm(x_sol - x2) / norm(x_sol);
fprintf("Erreur normwise sur le résidu de la matrice permutée : %4d \n", normwise_res);  
fprintf("Erreur normwise sur la solution : %4d \n", normwise_sol);  
fprintf("Nombre d'opérations pour la phase de résolution : %4d \n", 2*nnz(L) - 2*n + 2*nnz(U) - n); 
