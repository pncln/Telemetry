import React from 'react';
import HomeScreen from './screens/HomeScreen';
import SigninScreen from './screens/SigninScreen';
import DashScreen from './screens/DashScreen';

import { AuthProvider } from './contexts/AuthContext';
import { BrowserRouter, Route, Routes, Navigate } from 'react-router-dom'

const App = () => {
    return (
        <div className="root-container">
            <BrowserRouter>
            <AuthProvider>
                    <Routes>
                        {/* <Route index element={<HomeScreen />} /> */}
                        <Route index element={
                            <Navigate to="/signin" replace />
                        } />
                        <Route path="/signin" element={<SigninScreen />} />
                        <Route path="/dashboard" element={<DashScreen />} />
                        <Route
                            path="*"
                            element={<Navigate to="/" replace />}
                        />
                    </Routes>
            </AuthProvider>
            </BrowserRouter>
        </div>
    );
}

export default App;
