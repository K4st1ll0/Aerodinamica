close all, clear all,
clc


filename    = 'esfera1.stl';
[p,t,tnorm] = import_stl_fast(filename,1);

figure(2)
patch('vertices',p,'faces',t,'FaceColor','none','edgecolor','k')


U_infty = zeros(size(tnorm));
U_infty(:,2) = 1;
U_infty(:,1) = 0;

Stheta = zeros(length(tnorm),1);

for I=1:length(tnorm)

    Stheta(I,1) = dot(U_infty(I,:),tnorm(I,:))./norm(U_infty(I,:));
    
end

% Cálculo del cp. Método de Newton
cps = 2*Stheta.^2;
cps = [cps;0];

% Cálculo del cp en las zonas de sotavento
I_cpO           = find(Stheta>0);
cps(I_cpO,1)    = 0;


figure(1)
colormap(jet)
set(gca,'CLim',[0 2])
pat = patch('vertices',p,'faces',t,'FaceColor','b','edgecolor','none')
xlabel('x'),ylabel('y'),zlabel('z'),


set(gca,'CLim',[0 2])
cdata = cps;
set(pat,'FaceColor','flat','FaceVertexCData',cdata,'CDataMapping','scaled')
colorbar

% Longitud máxima necesaria
nmax = max([numel(cdata), numel(cps), numel(I_cpO), size(U_infty,1)]);

% Crear celda vacía para poder mezclar números y huecos
C = cell(nmax+1, 9);

% Encabezados
C(1,:) = {'cdata', '', 'cps', '', 'I_cpO', '', 'U_infty_x', 'U_infty_y', 'U_infty_z'};

% Rellenar columnas
for i = 1:numel(cdata)
    C{i+1,1} = cdata(i);
end

for i = 1:numel(cps)
    C{i+1,3} = cps(i);
end

for i = 1:numel(I_cpO)
    C{i+1,5} = I_cpO(i);
end

for i = 1:size(U_infty,1)
    C{i+1,7} = U_infty(i,1);
    C{i+1,8} = U_infty(i,2);
    C{i+1,9} = U_infty(i,3);
end

% Exportar a CSV
writecell(C, 'variables_workspace.csv');

theta = acos(Stheta);
theta_deg = rad2deg(theta);

cp_plot = cps(1:length(Stheta));

[theta_sorted, idx] = sort(theta_deg);
cp_sorted = cp_plot(idx);

figure
plot(theta_sorted, cp_sorted, '-o')
xlabel('\theta [deg]')
ylabel('c_p')
title('c_p vs \theta')
grid on
